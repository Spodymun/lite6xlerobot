#!/usr/bin/env python3
"""
Run ACT Policy + Robot Throw with Ball Detection, Pick, Cup Detection, and Direct Throw.

Workflow:
1) Find Ball
2) Pick Ball (top-down)
3) Go to INIT pose
4) Find Cup target
5) Go to cup position + open/close gripper
6) Run Policy inference to get throw trajectory
7) Execute throw via throw_from_job.py
8) Return to INIT pose
9) Done
"""

import argparse, json, subprocess, sys, os, time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

from xarm.wrapper import XArmAPI


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
    -1.570796,  # -90
     0.785398,  # 45
     2.356194,  # 135
     0.000000,  # 0
    -1.570796,  # -90
    -1.570796,  # -90
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


# ============================================================================
# BALL DETECTION
# ============================================================================
def parse_ball_line(line: str) -> Optional[BallDet]:
    line = line.strip()
    if not line.startswith("BALL_MM "):
        return None
    p = line.split()
    if len(p) < 4:
        return None
    try:
        return BallDet(float(p[1]), float(p[2]), float(p[3]))
    except ValueError:
        return None


def run_and_capture_first_ball(cmd, *, timeout_s: float, verbose: bool) -> Optional[BallDet]:
    start = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert p.stdout is not None
        for line in p.stdout:
            if verbose:
                print(line, end="")
            obj = parse_ball_line(line)
            if obj is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                return obj
            if time.time() - start > timeout_s:
                try:
                    p.kill()
                except Exception:
                    pass
                return None
        return None
    finally:
        try:
            p.kill()
        except Exception:
            pass


# ============================================================================
# CUP DETECTION
# ============================================================================
def parse_cup_line(line: str) -> Optional[CupDet]:
    line = line.strip()
    if not line.startswith("CUP_MM "):
        return None
    p = line.split()
    if len(p) < 4:
        return None
    try:
        return CupDet(float(p[1]), float(p[2]), float(p[3]))
    except ValueError:
        return None


def run_and_capture_first_cup(cmd, *, timeout_s: float, verbose: bool) -> Optional[CupDet]:
    start = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert p.stdout is not None
        for line in p.stdout:
            if verbose:
                print(line, end="")
            obj = parse_cup_line(line)
            if obj is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                return obj
            if time.time() - start > timeout_s:
                try:
                    p.kill()
                except Exception:
                    pass
                return None
        return None
    finally:
        try:
            p.kill()
        except Exception:
            pass


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
# ROBOT OPERATIONS
# ============================================================================
def arm_connect() -> XArmAPI:
    arm = XArmAPI(ROBOT_IP, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    return arm


def go_init_pose(arm: XArmAPI, *, speed: float, acc: float, wait: bool = True) -> bool:
    code = arm.set_servo_angle(
        angle=INIT_JOINTS_RAD,
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


def go_still_pose(arm: XArmAPI, *, speed: float, acc: float, wait: bool = True) -> bool:
    code = arm.set_servo_angle(
        angle=STILL_JOINTS_RAD,
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
    pick_y_offset_mm: float,
    pick_x_offset_mm: float,
) -> bool:
    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[list]:
        code, joints = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code == 0 and joints is not None:
            return [float(v) for v in joints[:6]]

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

    arm.open_lite6_gripper(sync=True)
    time.sleep(0.2)

    rx, ry = mapper.map(ball.x_mm, ball.y_mm)
    rx = rx + float(pick_x_offset_mm)
    ry = ry + float(pick_y_offset_mm)

    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False
    code = arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move hover failed code={code}")
        return False

    j_pick = ik_joints_for_pose_xyz_rpy_deg(rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False
    code = arm.set_servo_angle(angle=j_pick, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move down failed code={code}")
        return False

    time.sleep(0.25)
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.3)

    j_lift = ik_joints_for_pose_xyz_rpy_deg(rx, ry, lift_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_lift is None:
        print("[ERR] IK lift failed")
        return False
    code = arm.set_servo_angle(angle=j_lift, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] lift failed code={code}")
        return False

    return True


def go_cup_position_and_release(
    arm: XArmAPI,
    mapper: TableToRobot,
    cup: CupDet,
    *,
    hover_z_mm: float,
    ik_speed: float,
    ik_acc: float,
) -> bool:
    """Move to cup position, open/close gripper briefly."""
    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[list]:
        code, joints = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code == 0 and joints is not None:
            return [float(v) for v in joints[:6]]

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

    rx, ry = mapper.map(cup.x_mm, cup.y_mm)

    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK cup hover failed")
        return False
    code = arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move to cup hover failed code={code}")
        return False

    print("[GRIP] open (cup)...")
    arm.open_lite6_gripper(sync=True)
    time.sleep(1.0)

    print("[GRIP] close (cup)...")
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.3)

    return True


def lerp(a, b, t: float):
    """Linear interpolation between two same-length lists."""
    return [(1.0 - t) * av + t * bv for av, bv in zip(a, b)]


def fk_tcp_xyz_mm(robot_ip: str, joints_rad):
    """Returns [x,y,z] in mm for given joint angles (rad) using robot FK."""
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    try:
        try:
            code, pose = arm.get_forward_kinematics(joints_rad, is_radian=True)
        except TypeError:
            code, pose = arm.get_forward_kinematics(joints_rad)

        if code != 0 or pose is None:
            raise RuntimeError(f"FK failed: code={code}, pose={pose}")

        xyz = [float(pose[0]), float(pose[1]), float(pose[2])]
        return xyz
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def resolve_repo_root_and_paths(args):
    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve()

    ckpt_in = (repo_root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()

    if (ckpt_in / "pretrained_model").is_dir():
        ckpt_dir = ckpt_in / "pretrained_model"
    else:
        ckpt_dir = ckpt_in

    return repo_root, root, ckpt_dir


def load_policy_local(ckpt_dir: Path):
    """Load ACTPolicy from local directory."""
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in: {ckpt_dir}")

    cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    policy = ACTPolicy.from_pretrained(str(ckpt_dir), config=cfg)
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="records/throws/fs_2026-02-08_ALL_THROWS_FIXED")
    ap.add_argument("--ckpt", required=True,
                    help="Path to checkpoint dir or pretrained_model dir.")
    ap.add_argument("--seed_index", type=int, default=0, help="Dataset frame for state/env_state")
    ap.add_argument("--job_out", default="/tmp/wurf_job.json")
    ap.add_argument("--result_out", default="/tmp/throw_result.json")
    ap.add_argument("--robot_ip", default="10.77.77.200")
    ap.add_argument("--dry_run", action="store_true", help="Only compute, don't throw")

    # Ball/Cup detection
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ball-timeout-s", type=float, default=15.0)
    ap.add_argument("--cup-timeout-s", type=float, default=15.0)
    ap.add_argument("--verbose-detectors", action="store_true")

    # Mapping + pick
    ap.add_argument("--table-to-robot-yaml", type=str, default="calibrate/table_to_robot.yaml")
    ap.add_argument("--pick-y-offset-mm", type=float, default=0.0)
    ap.add_argument("--pick-x-offset-mm", type=float, default=0.0)
    ap.add_argument("--hover-z-mm", type=float, default=100.0)
    ap.add_argument("--pick-z-mm", type=float, default=-1.5)
    ap.add_argument("--lift-z-mm", type=float, default=120.0)
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # FK
    ap.add_argument("--no_fk", action="store_true", help="Skip FK computation (throw_from_job will fail if skipped)")

    ap.add_argument("--release_override", type=float, default=None)
    ap.add_argument("--release_bias", type=float, default=0.0)

    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    repo_root, root, ckpt_dir = resolve_repo_root_and_paths(args)

    cup = None  # Will be populated in STILL pose with open gripper

    # ========================================================================
    # STEP 1-2: FIND + PICK BALL
    # ========================================================================
    if not args.dry_run:
        arm = arm_connect()
        
        coord_dir = repo_root / "koordinaten"
        ball_script = coord_dir / "ball_finder.py"
        H_path = coord_dir / args.H if not Path(args.H).is_absolute() else Path(args.H)
        
        if not ball_script.exists():
            raise RuntimeError(f"ball_finder.py not found: {ball_script}")
        if not H_path.exists():
            raise RuntimeError(f"H file not found: {H_path}")

        ball_cmd = [
            sys.executable,
            str(ball_script),
            "--cam", str(args.cam),
            "--H", str(H_path),
            "--device", args.device,
            "--once",
        ]
        print("[BALL] Searching...")
        ball = run_and_capture_first_ball(ball_cmd, timeout_s=args.ball_timeout_s, verbose=args.verbose_detectors)
        if ball is None:
            raise RuntimeError("Ball not found (timeout)")
        print(f"[BALL] OK ({ball.x_mm:.1f},{ball.y_mm:.1f}) conf={ball.conf:.2f}")

        # Pick ball
        mapper = TableToRobot(os.path.expanduser(args.table_to_robot_yaml))
        print("[PICK] Picking...")
        ok = pick_ball_topdown(
            arm,
            mapper,
            ball,
            hover_z_mm=args.hover_z_mm,
            pick_z_mm=args.pick_z_mm,
            lift_z_mm=args.lift_z_mm,
            ik_speed=args.ik_speed,
            ik_acc=args.ik_acc,
            pick_y_offset_mm=args.pick_y_offset_mm,
            pick_x_offset_mm=args.pick_x_offset_mm,
        )
        if not ok:
            raise RuntimeError("Pick failed")
        print("[PICK] OK")

        # Go to STILL pose (NOT INIT - cup detection will run now)
        print("[POSE] going to STILL pose...")
        if not go_still_pose(arm, speed=args.ik_speed, acc=args.ik_acc, wait=True):
            raise RuntimeError("go_still_pose failed")
        print("[POSE] STILL reached")

        # ====================================================================
        # GRIPPER OPEN/CLOSE AT STILL POSE
        # ====================================================================
        print("[GRIP] open (still)...")
        arm.open_lite6_gripper(sync=True)

        # ====================================================================
        # STEP 3: DETECT CUP WITH OPEN GRIPPER (ball centering)
        # ====================================================================
        coord_dir = repo_root / "koordinaten"
        cups_script = coord_dir / "cups_yolo.py"
        H_path = coord_dir / args.H if not Path(args.H).is_absolute() else Path(args.H)
        H_fixed_npz = coord_dir / "H_fixed.npz"

        if not cups_script.exists():
            raise RuntimeError(f"cups_yolo.py not found: {cups_script}")

        cup_cmd = [
            sys.executable,
            str(cups_script),
            "--H", str(H_path),
            "--once",
        ]
        if H_fixed_npz.exists():
            cup_cmd.extend(["--warp_npz", str(H_fixed_npz)])

        print("[CUP] Searching for cup target (gripper open, ball centered)...")
        cup = run_and_capture_first_cup(cup_cmd, timeout_s=args.cup_timeout_s, verbose=args.verbose_detectors)
        if cup is None:
            raise RuntimeError("Cup not found (timeout) - needed as target for policy")
        print(f"[CUP] Target OK ({cup.x_mm:.1f},{cup.y_mm:.1f}) conf={cup.conf:.2f}")

        print("[GRIP] close (still)...")
        arm.close_lite6_gripper(sync=True)
        time.sleep(0.3)

        # Disconnect arm (robot will go to INIT before throw)
        try:
            arm.disconnect()
        except Exception:
            pass
    else:
        # Dry run: use origin as cup position
        cup = CupDet(0.0, 0.0, 1.0)
    
    # ========================================================================
    # STEP 5: RUN POLICY INFERENCE
    # ========================================================================
    ds = LeRobotDataset(repo_id="local/xarm_throws_fixed", root=str(root))
    policy = load_policy_local(ckpt_dir)

    x = ds[args.seed_index]

    # Use cup position as target
    if not args.dry_run:
        target_xy = torch.tensor([cup.x_mm, cup.y_mm], dtype=torch.float32)
    else:
        # For dry_run, use origin
        target_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)

    env = x["observation.environment_state"].float().clone()
    if env.numel() != 9:
        raise RuntimeError(f"expected env_state 9D, got {env.numel()}")

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

        if isinstance(out, tuple):
            actions_hat = out[0]
        else:
            actions_hat = out

        pred = actions_hat[0, 0].cpu() if actions_hat.dim() == 3 else actions_hat[0].cpu()

    if pred.numel() != 13:
        raise RuntimeError(f"Expected 13D action, got {pred.numel()}")

    pos1 = [float(v) for v in pred[0:6].tolist()]
    pos2 = [float(v) for v in pred[6:12].tolist()]
    release_at = float(pred[12].item())

    if args.release_override is not None:
        release_at = float(args.release_override)

    release_at = max(0.05, min(0.95, release_at + float(args.release_bias)))
    release_joints = lerp(pos1, pos2, release_at)

    do_fk = (not args.no_fk) and (not args.dry_run)
    release_xyz = None
    if do_fk:
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

    print("[OK] wrote job:", str(job_path))
    print("ckpt_dir:", str(ckpt_dir))
    print("target:", [float(target_xy[0].item()), float(target_xy[1].item())])
    print("pos1:", pos1)
    print("pos2:", pos2)
    print("release_at:", release_at)
    print("release_joints:", release_joints)
    if release_xyz is None:
        print("release_xyz(mm): <skipped>")
    else:
        print("release_xyz(mm):", release_xyz)

    if args.dry_run:
        print("[DONE] Dry run complete.")
        return

    # ========================================================================
    # STEP 6: EXECUTE THROW
    # ========================================================================
    print("[THROW] Executing throw...")
    throw_script = repo_root / "wuerfe" / "throw_from_job.py"
    cmd = [sys.executable, str(throw_script), "--job", str(job_path), "--result", args.result_out, "--no-init"]
    subprocess.run(cmd, check=True)
    print("[THROW] Done.")

    # ========================================================================
    # STEP 7: RETURN TO INIT
    # ========================================================================
    arm = arm_connect()
    print("[POSE] Returning to INIT...")
    if not go_init_pose(arm, speed=args.ik_speed, acc=args.ik_acc, wait=True):
        raise RuntimeError("Final go_init_pose failed")
    print("[POSE] INIT reached.")
    try:
        arm.disconnect()
    except Exception:
        pass
    print("[DONE] Workflow complete.")


if __name__ == "__main__":
    main()
