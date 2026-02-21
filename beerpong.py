#!/usr/bin/env python3
"""
Run ACT Policy + Robot Throw with Live Camera Ball/Cup Detection.

Workflow:
1) Find Ball (camera)  [BALL: no warp]
2) Pick Ball top-down  (using camera table-mm -> robot mapping)
3) Go to STILL pose
4) Find Cup target (camera) [CUP: warped if available]
5) Run ACT policy inference for this cup target
6) Execute throw via throw_from_job.py
7) Return to INIT pose
8) Loop
"""

import argparse, json, subprocess, sys, os, time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import threading

import torch
import numpy as np
import yaml
import cv2
from ultralytics import YOLO

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

from xarm.wrapper import XArmAPI

from calibrate.cup_warp import load_cup_warp_npz, apply_cup_warp

# ============================================================================
# CONSTANTS
# ============================================================================
ROBOT_IP = "10.77.77.200"

INIT_JOINTS_RAD = [
    -1.570796,
     0.717557,
     1.192005,
     0.000000,
     0.474205,
    -1.570796,
]

STILL_JOINTS_RAD = [
    -1.570796,
     0.785398,
     2.356194,
     0.000000,
    -1.570796,
    -1.570796,
]

FIX_R, FIX_P, FIX_YAW = -180, 0, 0


# ============================================================================
# DATACLASSES
# ============================================================================
@dataclass
class BallDet:
    x_mm: float
    y_mm: float
    conf: float


@dataclass
class CupDet:
    x_mm: float
    y_mm: float
    conf: float


def truncate_coord(val: float) -> float:
    """Truncate coordinate to 1 decimal place (e.g., 15.57 -> 15.5)."""
    return float(int(val * 10) / 10.0)


# ============================================================================
# TABLE TO ROBOT MAPPING
# ============================================================================
class TableToRobot:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        self.A = np.array(d["A"], dtype=float)
        self.b = np.array(d["b"], dtype=float)

    def map(self, x_table_mm: float, y_table_mm: float) -> Tuple[float, float]:
        p = np.array([x_table_mm, y_table_mm], dtype=float)
        r = self.A @ p + self.b
        return float(r[0]), float(r[1])


# ============================================================================
# LIVE CAMERA
# ============================================================================
class LiveCamera:
    """
    Threaded live camera:
    - A background thread continuously reads frames -> always fresh.
    - get_detections() runs YOLO on the latest frame.
    - Visualization always shows the latest frame (with optional boxes).
    """

    def __init__(
        self,
        cam_idx: int,
        H_path: str,
        weights_path: str,
        device: str = "cpu",
        ball_class: str = "ball",
        cup_class: str = "cup",
        conf_threshold: float = 0.35,
        cup_warp_npz: Optional[str] = None,
    ):
        self.cam_idx = cam_idx
        self.device = device
        self.ball_class = ball_class
        self.cup_class = cup_class
        self.conf_threshold = conf_threshold

        # Load homography
        h_path = Path(H_path)
        if not h_path.exists():
            raise RuntimeError(f"H file not found: {h_path.resolve()}")
        self.H_fixed = np.load(str(h_path))
        if self.H_fixed.shape != (3, 3):
            raise RuntimeError(f"H must be 3x3 .npy, got shape={self.H_fixed.shape}")

        # Load cup warp (optional)
        self.cup_warp = None
        if cup_warp_npz:
            wp = Path(cup_warp_npz)
            if wp.exists():
                self.cup_warp = load_cup_warp_npz(str(wp))

        # Load YOLO
        w = Path(weights_path)
        if not w.exists():
            raise RuntimeError(f"Weights not found: {w}")
        self.model = YOLO(str(w))
        if device == "cpu":
            self.model.to("cpu")

        # Open camera
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {cam_idx}")

        # Optional: reduce internal buffering (some backends honor it)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Thread state
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        # Warmup: wait briefly for first frame
        t0 = time.time()
        while self._latest_frame is None and time.time() - t0 < 2.0:
            time.sleep(0.01)

    def _reader_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame
            # keep loop responsive
            time.sleep(0.001)

    def pixel_to_table_xy(self, pixel_xy: np.ndarray) -> np.ndarray:
        pt = np.array([[pixel_xy]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_fixed)
        return out[0, 0]

    def best_box_for_class(self, result, class_name: str):
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None
        names = result.names
        best = None
        for b in result.boxes:
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls = str(names.get(cls_id, cls_id))
            if cls != class_name:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cand = (conf, x1, y1, x2, y2, cls)
            if best is None or conf > best[0]:
                best = cand
        return best

    def get_frame(self):
        """Return latest frame (copy) or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_detections(
        self,
        visualize: bool = False,
        cup_area_bounds: Optional[dict] = None,
    ) -> Tuple[Optional[BallDet], Optional[CupDet]]:
        frame = self.get_frame()
        if frame is None:
            return None, None

        # Run YOLO on latest frame
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        r0 = results[0] if results else None

        ball_det: Optional[BallDet] = None
        cup_det: Optional[CupDet] = None

        vis = frame if visualize else None

        # CUP (warp allowed)
        cup_box = self.best_box_for_class(r0, self.cup_class)
        if cup_box is not None:
            conf, x1, y1, x2, y2, _ = cup_box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cup_mm = self.pixel_to_table_xy(np.array([cx, cy], dtype=np.float32))
            cup_mm_w = cup_mm

            if self.cup_warp is not None:
                xw, yw = apply_cup_warp(float(cup_mm[0]), float(cup_mm[1]), self.cup_warp)
                cup_mm_w = np.array([xw, yw], dtype=np.float32)

            if cup_area_bounds is None or (
                cup_area_bounds["x_min"] <= cup_mm_w[0] <= cup_area_bounds["x_max"]
                and cup_area_bounds["y_min"] <= cup_mm_w[1] <= cup_area_bounds["y_max"]
            ):
                cup_det = CupDet(float(cup_mm_w[0]), float(cup_mm_w[1]), float(conf))

                if visualize and vis is not None:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
                    cv2.putText(
                        vis,
                        f"CUP {conf:.2f} mm({cup_mm_w[0]:.1f},{cup_mm_w[1]:.1f})",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        # BALL (NO warp)
        ball_box = self.best_box_for_class(r0, self.ball_class)
        if ball_box is not None:
            conf, x1, y1, x2, y2, _ = ball_box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            ball_mm = self.pixel_to_table_xy(np.array([cx, cy], dtype=np.float32))
            ball_det = BallDet(float(ball_mm[0]), float(ball_mm[1]), float(conf))

            if visualize and vis is not None:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
                cv2.putText(
                    vis,
                    f"BALL {conf:.2f} mm({ball_mm[0]:.1f},{ball_mm[1]:.1f})",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        if visualize and vis is not None:
            cv2.imshow("beerpong - camera view", vis)
            cv2.waitKey(1)

        return ball_det, cup_det

    def close(self):
        self._running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


# ============================================================================
# ROBOT OPS
# ============================================================================
def arm_connect(robot_ip: str) -> XArmAPI:
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    return arm


def go_pose_joints(arm: XArmAPI, joints_rad, *, speed: float, acc: float, wait: bool = True) -> bool:
    code = arm.set_servo_angle(
        angle=joints_rad,
        speed=speed,
        mvacc=acc,
        is_radian=True,
        wait=wait,
    )
    if code != 0:
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)
        return False
    return True


def ik_joints_for_pose_xyz_rpy_deg(arm: XArmAPI, x, y, z, r, p, yaw) -> Optional[list]:
    code, joints = arm.get_inverse_kinematics(
        [float(x), float(y), float(z), float(r), float(p), float(yaw)],
        input_is_radian=False,
        return_is_radian=True,
    )
    if code == 0 and joints is not None:
        return [float(v) for v in joints[:6]]

    # fallback for ±180° flip
    if abs(float(r)) == 180.0:
        r2 = 180.0 if float(r) < 0 else -180.0
        code2, joints2 = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r2), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code2 == 0 and joints2 is not None:
            return [float(v) for v in joints2[:6]]

    return None


def pick_ball_topdown(
    arm: XArmAPI,
    mapper: TableToRobot,
    ball: BallDet,
    *,
    hover_z_mm: float,
    pick_z_mm: float,
    lift_z_mm: float,
    ik_speed: float,
    ik_acc: float,
) -> bool:
    arm.open_lite6_gripper(sync=True)
    time.sleep(0.2)

    rx, ry = mapper.map(ball.x_mm, ball.y_mm)

    j_hover = ik_joints_for_pose_xyz_rpy_deg(arm, rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False
    if arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True) != 0:
        print("[ERR] move hover failed")
        return False

    j_pick = ik_joints_for_pose_xyz_rpy_deg(arm, rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False
    if arm.set_servo_angle(angle=j_pick, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True) != 0:
        print("[ERR] move down failed")
        return False

    time.sleep(0.25)
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.3)

    j_lift = ik_joints_for_pose_xyz_rpy_deg(arm, rx, ry, lift_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_lift is None:
        print("[ERR] IK lift failed")
        return False
    if arm.set_servo_angle(angle=j_lift, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True) != 0:
        print("[ERR] lift failed")
        return False

    return True


def lerp(a, b, t: float):
    return [(1.0 - t) * av + t * bv for av, bv in zip(a, b)]


def fk_tcp_xyz_mm(robot_ip: str, joints_rad):
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    try:
        try:
            code, pose = arm.get_forward_kinematics(joints_rad, is_radian=True)
        except TypeError:
            code, pose = arm.get_forward_kinematics(joints_rad)

        if code != 0 or pose is None:
            raise RuntimeError(f"FK failed: code={code}, pose={pose}")

        return [float(pose[0]), float(pose[1]), float(pose[2])]
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def resolve_repo_root_and_paths(args):
    repo_root = Path(__file__).resolve().parent
    root = (repo_root / args.root).resolve()

    ckpt_in = (repo_root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()
    ckpt_dir = ckpt_in / "pretrained_model" if (ckpt_in / "pretrained_model").is_dir() else ckpt_in
    return repo_root, root, ckpt_dir


def load_policy_local(ckpt_dir: Path):
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in: {ckpt_dir}")

    cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    policy = ACTPolicy.from_pretrained(str(ckpt_dir), config=cfg)
    policy.eval()
    return policy


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="records/throws/fs_TRAIN_READY")
    ap.add_argument("--ckpt", default="outputs/train/2026-02-21/13-48-03_gym_manipulator_act/checkpoints/007000")
    ap.add_argument("--seed_index", type=int, default=0)
    ap.add_argument("--job_out", default="/tmp/wurf_job.json")
    ap.add_argument("--result_out", default="/tmp/throw_result.json")
    ap.add_argument("--robot_ip", default=ROBOT_IP)

    # camera
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--weights", type=str,
                    default="/home/nelly/src/lite6xlerobot/ball_erkennung/ball_and_cups.v1i.yolov8/runs/detect/train2/weights/best.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ball-timeout-s", type=float, default=15.0)
    ap.add_argument("--cup-timeout-s", type=float, default=15.0)
    ap.add_argument("--ball-stability-checks", type=int, default=5)

    # mapping + pick
    ap.add_argument("--table-to-robot-yaml", type=str, default="calibrate/table_to_robot.yaml")
    ap.add_argument("--hover-z-mm", type=float, default=100.0)
    ap.add_argument("--pick-z-mm", type=float, default=-1.5)
    ap.add_argument("--lift-z-mm", type=float, default=120.0)
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # cup warp (optional, measured mm -> true mm)
    ap.add_argument("--warp-npz", type=str, default="",
                    help="Path to cup_warp .npz (cx/cy coeffs). "
                         "If empty, auto-detects koordinaten/H_fixed.npz or koordinaten/cup_warp.npz")

    # cup area bounds
    ap.add_argument("--cup-area-x-min", type=float, default=-100.0)
    ap.add_argument("--cup-area-x-max", type=float, default=600.0)
    ap.add_argument("--cup-area-y-min", type=float, default=-100.0)
    ap.add_argument("--cup-area-y-max", type=float, default=800.0)

    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    repo_root, root, ckpt_dir = resolve_repo_root_and_paths(args)

    # Load dataset + policy once
    ds = LeRobotDataset(repo_id="local/xarm_throws_fixed", root=str(root))
    x = ds[args.seed_index]
    policy = load_policy_local(ckpt_dir)

    mapper = TableToRobot(os.path.expanduser(args.table_to_robot_yaml))

    # homography + optional cup warp
    coord_dir = repo_root / "calibrate/"
    H_path = coord_dir / args.H if not Path(args.H).is_absolute() else Path(args.H)
    if not H_path.exists():
        raise RuntimeError(f"H file not found: {H_path}")

    # Determine cup warp path: explicit arg > auto-detect candidates
    if args.warp_npz:
        warp_candidate = Path(args.warp_npz)
        if not warp_candidate.is_absolute():
            warp_candidate = coord_dir / args.warp_npz
        if not warp_candidate.exists():
            raise RuntimeError(f"--warp-npz file not found: {warp_candidate.resolve()}")
        cup_warp_npz = str(warp_candidate)
        print(f"[WARP] Using explicit warp file: {cup_warp_npz}")
    else:
        # Auto-detect: try H_fixed.npz and cup_warp.npz
        _candidates = [coord_dir / "H_fixed.npz", coord_dir / "cup_warp.npz"]
        cup_warp_npz = None
        for _c in _candidates:
            if _c.exists():
                cup_warp_npz = str(_c)
                print(f"[WARP] Auto-detected warp file: {cup_warp_npz}")
                break
        if cup_warp_npz is None:
            print(f"[WARP] WARNING: No warp file found in {coord_dir} "
                  f"(checked: {[c.name for c in _candidates]}). "
                  "Cup coordinates will NOT be warped. "
                  "Pass --warp-npz <path> to enable.")

    camera = LiveCamera(
        cam_idx=args.cam,
        H_path=str(H_path),
        weights_path=args.weights,
        device=args.device,
        conf_threshold=0.35,
        cup_warp_npz=cup_warp_npz,
    )
    print("[CAMERA] Live camera initialized")

    cup_bounds = {
        "x_min": args.cup_area_x_min,
        "x_max": args.cup_area_x_max,
        "y_min": args.cup_area_y_min,
        "y_max": args.cup_area_y_max,
    }

    throw_count = 0

    try:
        while True:
            throw_count += 1
            print(f"\n{'='*70}\nTHROW #{throw_count}\n{'='*70}\n")

            arm = arm_connect(args.robot_ip)

            # ============================================================
            # 1) FIND BALL (stable)
            # ============================================================
            print("[BALL] Waiting for stable detection...")
            ball: Optional[BallDet] = None
            start = time.time()
            stable = 0
            last_tx = last_ty = None

            while time.time() - start < args.ball_timeout_s:
                ball_det, _ = camera.get_detections(visualize=args.show_camera)
                if ball_det is None:
                    stable = 0
                    time.sleep(0.05)
                    continue

                tx, ty = truncate_coord(ball_det.x_mm), truncate_coord(ball_det.y_mm)

                if (last_tx == tx and last_ty == ty):
                    stable += 1
                else:
                    stable = 1
                    last_tx, last_ty = tx, ty

                print(f"[BALL] {stable}/{args.ball_stability_checks} @ ({tx:.1f},{ty:.1f}) conf={ball_det.conf:.2f}")

                if stable >= args.ball_stability_checks:
                    ball = ball_det
                    break

                time.sleep(0.05)

            if ball is None:
                print("[ERR] Ball not found/stable. Skipping throw.")
                try:
                    arm.disconnect()
                except Exception:
                    pass
                continue

            print(f"[BALL] OK ({ball.x_mm:.1f},{ball.y_mm:.1f}) conf={ball.conf:.2f}")

            # ============================================================
            # 2) PICK BALL
            # ============================================================
            print("[PICK] Picking ball...")
            if not pick_ball_topdown(
                arm,
                mapper,
                ball,
                hover_z_mm=args.hover_z_mm,
                pick_z_mm=args.pick_z_mm,
                lift_z_mm=args.lift_z_mm,
                ik_speed=args.ik_speed,
                ik_acc=args.ik_acc,
            ):
                print("[ERR] Pick failed. Skipping throw.")
                try:
                    arm.disconnect()
                except Exception:
                    pass
                continue
            print("[PICK] OK")

            # ============================================================
            # 3) GO STILL
            # ============================================================
            print("[POSE] Going to STILL...")
            if not go_pose_joints(arm, STILL_JOINTS_RAD, speed=args.ik_speed, acc=args.ik_acc, wait=True):
                print("[ERR] STILL pose failed. Skipping throw.")
                try:
                    arm.disconnect()
                except Exception:
                    pass
                continue

            # Greifer einmal auf und nach 3 Sekunden wieder zu (in STILL Pose)
            arm.open_lite6_gripper(sync=True)
            time.sleep(3.0)
            arm.close_lite6_gripper(sync=True)
            time.sleep(0.2)

            # ============================================================
            # 4) FIND CUP TARGET (within bounds)
            # ============================================================
            print("[CUP] Searching for cup in valid area...")
            target_cup: Optional[CupDet] = None
            start = time.time()

            while time.time() - start < args.cup_timeout_s:
                _, cup_det = camera.get_detections(visualize=args.show_camera, cup_area_bounds=cup_bounds)
                if cup_det is not None:
                    target_cup = cup_det
                    break
                time.sleep(0.05)

            if target_cup is None:
                print("[CUP] No valid cup found. Stopping loop.")
                try:
                    arm.disconnect()
                except Exception:
                    pass
                break

            print(f"[CUP] Target ({target_cup.x_mm:.1f},{target_cup.y_mm:.1f}) conf={target_cup.conf:.2f}")

            # disconnect before inference/throw script (keeps it clean)
            try:
                arm.disconnect()
            except Exception:
                pass

            # ============================================================
            # 5) POLICY INFERENCE
            # ============================================================
            target_xy = torch.tensor([target_cup.x_mm, target_cup.y_mm], dtype=torch.float32)

            env = x["observation.environment_state"].float().clone()
            if env.numel() != 9:
                raise RuntimeError(f"expected env_state 9D, got {env.numel()}")

            # keep your existing convention
            env[4] = target_xy[0]
            env[5] = target_xy[1]
            env[7] = target_xy[0]
            env[8] = target_xy[1]

            env_b = env.unsqueeze(0)
            state_b = x["observation.state"].float().unsqueeze(0)

            with torch.no_grad():
                out = policy.model({
                    "observation.state": state_b,
                    "observation.environment_state": env_b
                })
                actions_hat = out[0] if isinstance(out, tuple) else out
                pred = actions_hat[0, 0].cpu() if actions_hat.dim() == 3 else actions_hat[0].cpu()

            if pred.numel() != 13:
                raise RuntimeError(f"Expected 13D action, got {pred.numel()}")

            pos1 = [float(v) for v in pred[0:6].tolist()]
            pos2 = [float(v) for v in pred[6:12].tolist()]
            release_at = float(pred[12].item())

            release_at = max(0.69, min(0.89, release_at))


            release_joints = lerp(pos1, pos2, release_at)
            release_xyz = fk_tcp_xyz_mm(args.robot_ip, release_joints)

            job = {
                "schema": "wurf/v4",
                "ip": args.robot_ip,
                "pos1": pos1,
                "pos2": pos2,
                "release_progress": release_at,
                "xyz_tolerance_mm": 3.0,
                "release_xyz": release_xyz,
                "target_xy_mm": [float(target_xy[0].item()), float(target_xy[1].item())],
            }

            job_path = Path(args.job_out)
            job_path.write_text(json.dumps(job, indent=2))

            print("[JOB] created")
            print(" target:", job["target_xy_mm"])
            print(" release_at:", release_at)
            print(" release_xyz(mm):", release_xyz)

            # ============================================================
            # 6) EXECUTE THROW
            # ============================================================
            print("[THROW] Executing throw...")
            throw_script = repo_root / "wuerfe" / "throw_from_job.py"
            cmd = [sys.executable, str(throw_script), "--job", str(job_path), "--result", args.result_out, "--no-init"]
            subprocess.run(cmd, check=True)
            print("[THROW] Complete")

            # ============================================================
            # 7) RETURN INIT
            # ============================================================
            arm = arm_connect(args.robot_ip)
            print("[POSE] Returning to INIT...")
            if not go_pose_joints(arm, INIT_JOINTS_RAD, speed=args.ik_speed, acc=args.ik_acc, wait=True):
                print("[WARN] INIT pose failed, continuing.")
            try:
                arm.disconnect()
            except Exception:
                pass

            print(f"[OK] Throw #{throw_count} complete.\n")

    finally:
        print("[CLEANUP] closing camera...")
        try:
            camera.close()
        except Exception:
            pass

    print("[DONE] Game workflow complete.")


if __name__ == "__main__":
    main()