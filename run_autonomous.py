#!/usr/bin/env python3
"""
run_autonomous.py (gohome + init-pose + top-down pick)

Flow (as requested):

Episode start:
  go_home()          # optional reset (controller home)

Pick:
  Ball finden
  Top-down pick (fixed RPY, only Z changes: hover_z -> pick_z -> lift_z)
  go_init_pose()     # move to INIT pose (fixed joints from screenshot)

Throw:
  execute_wurf()     # uses wurf_<n>.py (POS1 + throw) and release mode

After throw:
  go_home()          # reset for next episode

Notes:
- go_home() uses controller home pose: arm.move_gohome()
- init pose is FIXED JOINTS in radians (from your screenshot).
- Default pick_z_mm is 15.5 (instead of 25.0).
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from xarm.wrapper import XArmAPI
except Exception:
    XArmAPI = None


# =========================
# Fixed config (requested)
# =========================

# INIT pose (joints) from your screenshot:
# J1 -90°, J2 41.1°, J3 68.3°, J4 -5.8°, J5 28°, J6 -90°
INIT_JOINTS_RAD = [
    -1.5707963267948966,
     0.7173303225696694,
     1.1920598791121269,
    -0.10122909661567112,
     0.4886921905584123,
    -1.5707963267948966,
]

# Fixed TOP-DOWN orientation for picking (degrees) – matches your previous calibration screenshots
FIX_R, FIX_P, FIX_YAW = -178.6, 2.1, 7.3


# =========================
# Dataclasses
# =========================
@dataclass
class BallDet:
    x_mm: float
    y_mm: float
    conf: float


# =========================
# Parsing helpers (detectors)
# =========================
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


def mm_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# =========================
# Subprocess helper
# =========================
def run_and_capture_first_match(
    cmd: List[str],
    *,
    timeout_s: float,
    parse_fn: Callable[[str], Optional[Any]],
    verbose: bool = False,
    cwd: Optional[str] = None,
) -> Tuple[Optional[Any], int]:
    start = time.time()

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
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
            obj = parse_fn(line)
            if obj is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                return obj, p.returncode or 0

            if time.time() - start > timeout_s:
                try:
                    p.kill()
                except Exception:
                    pass
                return None, 124

        return None, p.wait(timeout=1.0)

    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        raise


# =========================
# Table -> Robot mapping
# =========================
class TableToRobot:
    """
    Affine mapping:
      r = A @ p + b
    where p is (x_table_mm, y_table_mm)
    """

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        self.A = np.array(d["A"], dtype=float)
        self.b = np.array(d["b"], dtype=float)

    def map(self, x_table_mm: float, y_table_mm: float) -> Tuple[float, float]:
        p = np.array([x_table_mm, y_table_mm], dtype=float)
        r = self.A @ p + self.b
        return float(r[0]), float(r[1])


# =========================
# Episode recorder (npz) - optional
# =========================
class EpisodeRecorder:
    """
    Records a single episode (trajectory) into arrays, then saves .npz.

    obs_t:  [ball_x, ball_y, target_x, target_y, dx, dy]  (float32)
    state_t:
        joint angles rad (6) +
        tcp x,y,z, roll,pitch,yaw (6) (mm/deg)
    action_t:
        desired joint angles rad (6) + gripper_cmd (1)
        gripper_cmd: 0.0=open, 1.0=close
    """

    def __init__(self, hz: float):
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self._t0 = None

        self.obs: List[np.ndarray] = []
        self.state: List[np.ndarray] = []
        self.action: List[np.ndarray] = []
        self.t: List[float] = []

        self.last_joint_target = np.zeros((6,), dtype=np.float32)
        self.last_gripper = np.float32(0.0)

        self.ball_xy = None
        self.target_xy = None

    def start(self, *, ball_xy: Tuple[float, float], target_xy: Tuple[float, float]):
        self._t0 = time.time()
        self.obs.clear()
        self.state.clear()
        self.action.clear()
        self.t.clear()
        self.ball_xy = (float(ball_xy[0]), float(ball_xy[1]))
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def set_action(
        self,
        *,
        joint_target_rad: Optional[List[float]] = None,
        gripper: Optional[float] = None,
    ):
        if joint_target_rad is not None:
            jt = np.array(joint_target_rad[:6], dtype=np.float32)
            if jt.shape != (6,):
                raise ValueError("joint_target_rad must have 6 values")
            self.last_joint_target = jt
        if gripper is not None:
            self.last_gripper = np.float32(gripper)

    def _make_obs(self) -> np.ndarray:
        bx, by = self.ball_xy
        tx, ty = self.target_xy
        dx = tx - bx
        dy = ty - by
        return np.array([bx, by, tx, ty, dx, dy], dtype=np.float32)

    def step(self, *, joint_angles_rad: List[float], tcp_pose6: List[float]):
        if self._t0 is None:
            raise RuntimeError("EpisodeRecorder.start() must be called first")
        now = time.time()
        tt = float(now - self._t0)

        obs = self._make_obs()
        joints = np.array(joint_angles_rad[:6], dtype=np.float32)
        tcp = np.array(tcp_pose6[:6], dtype=np.float32)
        state = np.concatenate([joints, tcp], axis=0)

        action = np.concatenate(
            [self.last_joint_target, np.array([self.last_gripper], dtype=np.float32)],
            axis=0,
        )

        self.t.append(tt)
        self.obs.append(obs)
        self.state.append(state)
        self.action.append(action)

    def save(self, path: Path, *, meta: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            t=np.array(self.t, dtype=np.float32),
            obs=np.stack(self.obs, axis=0) if self.obs else np.zeros((0, 6), dtype=np.float32),
            state=np.stack(self.state, axis=0) if self.state else np.zeros((0, 12), dtype=np.float32),
            action=np.stack(self.action, axis=0) if self.action else np.zeros((0, 7), dtype=np.float32),
            meta=np.array([meta], dtype=object),
        )


# =========================
# Robot wrapper
# =========================
class Robot:
    def __init__(self, ip: str, *, is_radian: bool = True):
        if XArmAPI is None:
            raise RuntimeError("xarm.wrapper not available. Install xArm SDK in your env.")
        self.arm = XArmAPI(ip, is_radian=is_radian)
        self.arm.connect()
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.2)

    def disconnect(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass

    def clean_and_ready(self):
        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.set_state(0)
        except Exception:
            pass

    def get_joints_rad(self) -> List[float]:
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0 or angles is None:
            return [0.0] * 6
        return [float(x) for x in angles[:6]]

    def get_tcp_pose6(self) -> List[float]:
        code, pose = self.arm.get_position(is_radian=False)  # mm/deg
        if code != 0 or pose is None:
            return [0.0] * 6
        return [float(x) for x in pose[:6]]

    def set_joint_target(self, joint_target_rad: List[float], *, speed: float, acc: float, wait: bool):
        return self.arm.set_servo_angle(
            angle=joint_target_rad[:6],
            speed=speed,
            mvacc=acc,
            is_radian=True,
            wait=wait,
        )

    def open_gripper(self, sync: bool = False):
        return self.arm.open_lite6_gripper(sync=sync)

    def close_gripper(self, sync: bool = True):
        return self.arm.close_lite6_gripper(sync=sync)


# =========================
# Pose helpers
# =========================
def _check(robot: Robot, code: int, tag: str) -> bool:
    if code == 0:
        return True
    print(f"[ERR] {tag} failed code={code} -> clean_error + abort")
    robot.clean_and_ready()
    return False


def go_home(robot: Robot, *, wait: bool = True) -> bool:
    """Controller-Home (what you set on the robot/controller)."""
    code = robot.arm.move_gohome(wait=wait)
    return _check(robot, code, "move_gohome")


def go_init_pose(robot: Robot, *, speed: float, acc: float, wait: bool = True) -> bool:
    """Fixed init pose for throw (NOT controller home)."""
    code = robot.set_joint_target(INIT_JOINTS_RAD, speed=speed, acc=acc, wait=wait)
    return _check(robot, code, "go_init_pose")


# =========================
# Pick (Top-down): X/Y fixed, only Z changes, grip at pick_z
# =========================
def pick_ball_topdown(
    robot: Robot,
    rec: Optional[EpisodeRecorder],
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
    sample_hz: float,
) -> bool:
    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[List[float]]:
        code, joints = robot.arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code != 0 or joints is None:
            return None
        return [float(v) for v in joints[:6]]

    # open gripper before pick
    if rec:
        rec.set_action(gripper=0.0)
    robot.open_gripper(sync=False)
    time.sleep(0.05)

    # table -> robot XY
    rx, ry = mapper.map(ball.x_mm, ball.y_mm)
    rx = rx + float(pick_x_offset_mm)
    ry = ry + float(pick_y_offset_mm)

    # 1) above ball
    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False
    if rec:
        rec.set_action(joint_target_rad=j_hover, gripper=0.0)
    if not _check(robot, robot.set_joint_target(j_hover, speed=ik_speed, acc=ik_acc, wait=True), "pick: move hover"):
        return False

    # 2) straight down to pick z
    j_pick = ik_joints_for_pose_xyz_rpy_deg(rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False
    if rec:
        rec.set_action(joint_target_rad=j_pick, gripper=0.0)
    if not _check(robot, robot.set_joint_target(j_pick, speed=ik_speed, acc=ik_acc, wait=True), "pick: move down"):
        return False

    # 3) close gripper
    if rec:
        rec.set_action(gripper=1.0)
    robot.close_gripper(sync=True)
    time.sleep(0.08)

    # 4) lift straight up
    j_lift = ik_joints_for_pose_xyz_rpy_deg(rx, ry, lift_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_lift is None:
        print("[ERR] IK lift failed")
        return False
    if rec:
        rec.set_action(joint_target_rad=j_lift, gripper=1.0)
    if not _check(robot, robot.set_joint_target(j_lift, speed=ik_speed, acc=ik_acc, wait=True), "pick: lift"):
        return False

    # short tail record
    if rec:
        t0 = time.time()
        while time.time() - t0 < 0.25:
            rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())
            time.sleep(1.0 / sample_hz)

    return True


# =========================
# Load & execute Wurfskript
# =========================
def import_wurf_module(wurf_path: Path):
    spec = importlib.util.spec_from_file_location(wurf_path.stem, str(wurf_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {wurf_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_wurf_target_xy_mm(mod) -> Tuple[float, float]:
    t = getattr(mod, "TARGET_XY_MM", None)
    if t is None:
        return (float("nan"), float("nan"))
    try:
        return (float(t[0]), float(t[1]))
    except Exception:
        return (float("nan"), float("nan"))


def execute_wurf(
    robot: Robot,
    rec: Optional[EpisodeRecorder],
    mod,
    *,
    positions_file: Optional[str],
    mode: str,
    pos1_speed: float,
    pos1_acc: float,
    throw_speed: float,
    throw_acc: float,
    open_after: float,
    release_at: float,
    poll_dt: float,
    progress_timeout: float,
    release_xyz: Optional[Tuple[float, float, float]],
    tol_mm: float,
    tcp_timeout: float,
    tail_s: float,
    record_hz: float,
):
    if not hasattr(mod, "load_positions"):
        raise ValueError("Wurfskript must define load_positions(positions_file)")

    start_pos, end_pos = mod.load_positions(positions_file)

    # Move to POS1
    if rec:
        rec.set_action(joint_target_rad=start_pos, gripper=1.0)
    _check(robot, robot.set_joint_target(start_pos, speed=pos1_speed, acc=pos1_acc, wait=True), "wurf: move POS1")

    # Throw command
    t0 = time.time()
    if rec:
        rec.set_action(joint_target_rad=end_pos, gripper=1.0)
    _check(robot, robot.set_joint_target(end_pos, speed=throw_speed, acc=throw_acc, wait=False), "wurf: throw cmd")

    if mode == "queue":
        while time.time() - t0 < max(0.0, open_after):
            if rec:
                rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())
            time.sleep(1.0 / record_hz)
        if rec:
            rec.set_action(gripper=0.0)
        robot.open_gripper(sync=False)

    elif mode == "progress":
        joint_progress = getattr(mod, "joint_progress", None)
        if joint_progress is None:
            raise ValueError("Wurfskript missing joint_progress() needed for mode=progress")

        code, start_angles = robot.arm.get_servo_angle(is_radian=True)
        if code != 0 or start_angles is None:
            start_angles = start_pos
        start_angles = start_angles[:6]

        opened = False
        while (time.time() - t0) < progress_timeout and not opened:
            c, cur = robot.arm.get_servo_angle(is_radian=True)
            if c == 0 and cur is not None:
                p = float(joint_progress(cur, start_angles, end_pos))
                if p >= float(release_at):
                    if rec:
                        rec.set_action(gripper=0.0)
                    robot.open_gripper(sync=False)
                    opened = True
                    break
            if rec:
                rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())
            time.sleep(max(0.001, float(poll_dt)))

        if not opened:
            if rec:
                rec.set_action(gripper=0.0)
            robot.open_gripper(sync=False)

    else:  # tcp
        rx, ry, rz = release_xyz if release_xyz is not None else (None, None, None)

        if rx is None or ry is None or rz is None:
            # WARNING: if not provided, defaults to current TCP -> triggers quickly
            code, pos = robot.arm.get_position(is_radian=False)
            if code == 0 and pos is not None:
                rx, ry, rz = float(pos[0]), float(pos[1]), float(pos[2])
            else:
                rx, ry, rz = 0.0, 0.0, 0.0

        tol2 = float(tol_mm) * float(tol_mm)
        opened = False
        while (time.time() - t0) < tcp_timeout and not opened:
            code, pos = robot.arm.get_position(is_radian=False)
            if code == 0 and pos is not None:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                dx, dy, dz = x - rx, y - ry, z - rz
                if (dx * dx + dy * dy + dz * dz) <= tol2:
                    if rec:
                        rec.set_action(gripper=0.0)
                    robot.open_gripper(sync=False)
                    opened = True
                    break
            if rec:
                rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())
            time.sleep(max(0.001, float(poll_dt)))

        if not opened:
            if rec:
                rec.set_action(gripper=0.0)
            robot.open_gripper(sync=False)

    # Tail sampling after release
    t_tail = time.time()
    while time.time() - t_tail < float(tail_s):
        if rec:
            rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())
        time.sleep(1.0 / record_hz)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    # robot & detector
    ap.add_argument("--ip", type=str, required=True, help="Robot IP, e.g. 10.77.77.200")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--verbose-detectors", action="store_true")

    # pick calibration mapping
    ap.add_argument(
        "--table-to-robot-yaml",
        type=str,
        default="/home/nelly/src/lite6xlerobot/calibrate/table_to_robot.yaml",
    )
    ap.add_argument("--pick-y-offset-mm", type=float, default=-30.0)
    ap.add_argument("--pick-x-offset-mm", type=float, default=3.0)

    # pick params (mm)
    ap.add_argument("--hover-z-mm", type=float, default=100.0, help="Z above ball before moving down")
    ap.add_argument("--pick-z-mm", type=float, default=15.5, help="Z for gripping (requested default=15.5)")
    ap.add_argument("--lift-z-mm", type=float, default=120.0, help="Z after gripping (lift)")
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # init pose motion params
    ap.add_argument("--init-speed", type=float, default=1.0)
    ap.add_argument("--init-acc", type=float, default=1.0)

    # home behavior (fixed to your requested flow)
    ap.add_argument("--skip-home-start", action="store_true", help="Skip go_home() at episode start")
    ap.add_argument("--skip-home-after-throw", action="store_true", help="Skip go_home() after throw")

    # wurf selection
    ap.add_argument("--wurf", type=int, required=True, help="Wurf-Nummer -> wuerfe/wurf_<n>.py")
    ap.add_argument("--wurf-dir", type=str, default="/home/nelly/src/lite6xlerobot/wuerfe")
    ap.add_argument("--wurf-positions", type=str, default=None, help="Optional positions file passed to load_positions()")
    ap.add_argument("--wurf-mode", choices=["queue", "progress", "tcp"], default="tcp")

    # speeds/acc (POS1 + throw)
    ap.add_argument("--pos1-speed", type=float, default=1.0)
    ap.add_argument("--pos1-acc", type=float, default=1.0)
    ap.add_argument("--throw-speed", type=float, default=3.0)
    ap.add_argument("--throw-acc", type=float, default=6.0)

    # release params
    ap.add_argument("--open-after", type=float, default=0.06)
    ap.add_argument("--release-at", type=float, default=0.35)
    ap.add_argument("--poll-dt", type=float, default=0.005)
    ap.add_argument("--progress-timeout", type=float, default=2.0)

    ap.add_argument("--release-x", type=float, default=None)
    ap.add_argument("--release-y", type=float, default=None)
    ap.add_argument("--release-z", type=float, default=None)
    ap.add_argument("--tol-mm", type=float, default=20.0)
    ap.add_argument("--tcp-timeout", type=float, default=2.0)

    # dataset recording
    ap.add_argument("--dataset-dir", type=str, default=None, help="If set, saves episode_XXXXXX.npz here")
    ap.add_argument("--record-hz", type=float, default=30.0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--min-steps", type=int, default=5)

    # loop control
    ap.add_argument("--max-throws", type=int, default=1)

    # ball return logic
    ap.add_argument("--ball-return-y", type=float, default=500.0)
    ap.add_argument("--ball-return-min-conf", type=float, default=0.4)
    ap.add_argument("--ball-stable-n", type=int, default=5)
    ap.add_argument("--ball-stable-eps-mm", type=float, default=8.0)
    ap.add_argument("--ball-return-poll", type=float, default=0.6)
    ap.add_argument("--ball-return-timeout", type=float, default=120.0)

    args = ap.parse_args()

    # paths
    base = Path(__file__).resolve().parent
    workdir = base / "koordinaten"
    ball_script = workdir / "ball_finder.py"
    H_path = workdir / args.H

    if not ball_script.exists():
        raise RuntimeError(f"ball_finder.py not found at {ball_script}")
    if not H_path.exists():
        raise RuntimeError(f"H.npy not found at {H_path}")

    # detector cmd
    ball_cmd = [
        sys.executable,
        str(ball_script),
        "--cam",
        str(args.cam),
        "--H",
        str(H_path),
        "--device",
        args.device,
        "--once",
    ]

    # mapping yaml
    mapper = TableToRobot(args.table_to_robot_yaml)

    # load wurf module
    wurf_path = Path(args.wurf_dir) / f"wurf_{args.wurf}.py"
    if not wurf_path.exists():
        raise FileNotFoundError(f"Wurfskript nicht gefunden: {wurf_path}")
    wurf_mod = import_wurf_module(wurf_path)
    target_xy = get_wurf_target_xy_mm(wurf_mod)

    # recording
    rec: Optional[EpisodeRecorder] = None
    dataset_dir: Optional[Path] = None
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rec = EpisodeRecorder(hz=args.record_hz)

    # robot
    robot = Robot(args.ip, is_radian=True)

    print("[RUN] Autonomous started (pick ball + wurf script throw)")
    print(f"[RUN] WURF: {wurf_path}  mode={args.wurf_mode}")
    print(f"[RUN] INIT_JOINTS(rad): {', '.join(f'{v:.3f}' for v in INIT_JOINTS_RAD)}")
    print(f"[RUN] PICK Z defaults: hover_z={args.hover_z_mm}  pick_z={args.pick_z_mm}  lift_z={args.lift_z_mm}")

    throws = 0
    episode_idx = 0

    try:
        while True:
            if args.max_throws > 0 and throws >= args.max_throws:
                print("[RUN] Done")
                break

            # ---- EPISODE START: go_home() ----
            robot.open_gripper(sync=True)
            if not args.skip_home_start:
                if not go_home(robot, wait=True):
                    print("[WARN] go_home failed at episode start")
                    continue

            # ---- BALL ----
            ball, _ = run_and_capture_first_match(
                ball_cmd,
                timeout_s=15.0,
                parse_fn=parse_ball_line,
                verbose=args.verbose_detectors,
                cwd=str(workdir),
            )
            if ball is None:
                print("[WARN] Ball not found")
                continue

            print(f"[OK] BALL ({ball.x_mm:.1f},{ball.y_mm:.1f}) conf={ball.conf:.2f}")

            # ---- RECORD START ----
            if rec:
                rec.start(ball_xy=(ball.x_mm, ball.y_mm), target_xy=target_xy)
                rec.set_action(joint_target_rad=robot.get_joints_rad(), gripper=0.0)
                rec.step(joint_angles_rad=robot.get_joints_rad(), tcp_pose6=robot.get_tcp_pose6())

            # ---- PICK (TOP-DOWN) ----
            ok = pick_ball_topdown(
                robot,
                rec,
                mapper,
                ball,
                hover_z_mm=args.hover_z_mm,
                pick_z_mm=args.pick_z_mm,
                lift_z_mm=args.lift_z_mm,
                ik_speed=args.ik_speed,
                ik_acc=args.ik_acc,
                pick_y_offset_mm=args.pick_y_offset_mm,
                pick_x_offset_mm=args.pick_x_offset_mm,
                sample_hz=args.record_hz,
            )
            if not ok:
                print("[WARN] pick failed")
                continue

            # ---- AFTER PICK: go_init_pose() (requested) ----
            if not go_init_pose(robot, speed=args.init_speed, acc=args.init_acc, wait=True):
                print("[WARN] go_init_pose failed after pick")
                continue

            # ---- THROW ----
            release_xyz = None
            if args.release_x is not None and args.release_y is not None and args.release_z is not None:
                release_xyz = (args.release_x, args.release_y, args.release_z)

            execute_wurf(
                robot,
                rec,
                wurf_mod,
                positions_file=args.wurf_positions,
                mode=args.wurf_mode,
                pos1_speed=args.pos1_speed,
                pos1_acc=args.pos1_acc,
                throw_speed=args.throw_speed,
                throw_acc=args.throw_acc,
                open_after=args.open_after,
                release_at=args.release_at,
                poll_dt=args.poll_dt,
                progress_timeout=args.progress_timeout,
                release_xyz=release_xyz,
                tol_mm=args.tol_mm,
                tcp_timeout=args.tcp_timeout,
                tail_s=args.tail_s,
                record_hz=args.record_hz,
            )

            # ---- AFTER THROW: go_home() ----
            if not args.skip_home_after_throw:
                go_home(robot, wait=True)

            # ---- SAVE ----
            if rec and dataset_dir is not None:
                if len(rec.t) < args.min_steps:
                    print(f"[WARN] Episode too short ({len(rec.t)} steps), skipping save.")
                else:
                    episode_idx += 1
                    ep_path = dataset_dir / f"episode_{episode_idx:06d}.npz"
                    rec.save(
                        ep_path,
                        meta={
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "ball": {"x": ball.x_mm, "y": ball.y_mm, "conf": ball.conf},
                            "target_xy_mm": [float(target_xy[0]), float(target_xy[1])],
                            "wurf_script": str(wurf_path),
                            "wurf_mode": args.wurf_mode,
                            "wurf_positions": args.wurf_positions,
                            "init_joints_rad": [float(v) for v in INIT_JOINTS_RAD],
                            "pick": {
                                "hover_z_mm": args.hover_z_mm,
                                "pick_z_mm": args.pick_z_mm,
                                "lift_z_mm": args.lift_z_mm,
                                "pick_y_offset_mm": args.pick_y_offset_mm,
                                "pick_x_offset_mm": args.pick_x_offset_mm,
                                "ik_speed": args.ik_speed,
                                "ik_acc": args.ik_acc,
                                "topdown_rpy_deg": [FIX_R, FIX_P, FIX_YAW],
                            },
                        },
                    )
                    print(f"[SAVE] {ep_path}  steps={len(rec.t)}")

            throws += 1

            # ---- WAIT FOR BALL RETURN (STABLE) ----
            print("[WAIT] Waiting for stable ball return...")
            t0 = time.time()
            stable = 0
            last_xy = None

            while True:
                if time.time() - t0 > args.ball_return_timeout:
                    print("[WAIT] Timeout, continue anyway")
                    break

                bd, _ = run_and_capture_first_match(
                    ball_cmd,
                    timeout_s=10.0,
                    parse_fn=parse_ball_line,
                    verbose=False,
                    cwd=str(workdir),
                )

                if bd is None:
                    time.sleep(args.ball_return_poll)
                    continue

                if bd.conf < args.ball_return_min_conf or bd.y_mm <= args.ball_return_y:
                    stable = 0
                    last_xy = None
                    time.sleep(args.ball_return_poll)
                    continue

                cur = (bd.x_mm, bd.y_mm)
                if last_xy is None or mm_dist(cur, last_xy) > args.ball_stable_eps_mm:
                    stable = 1
                    last_xy = cur
                else:
                    stable += 1

                print(f"[WAIT] stable {stable}/{args.ball_stable_n} at ({bd.x_mm:.1f},{bd.y_mm:.1f})")

                if stable >= args.ball_stable_n:
                    print("[WAIT] Ball is stable")
                    break

                time.sleep(args.ball_return_poll)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()