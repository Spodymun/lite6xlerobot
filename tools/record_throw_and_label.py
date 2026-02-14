#!/usr/bin/env python3
"""
tools/record_throw_and_label.py

Ablauf:
1) Ball finden (koordinaten/ball_finder.py -> "BALL_MM x y conf")
2) Ball aufnehmen (Top-Down Pick, IK, fixed RPY)
3) STILL Pose anfahren (fest im Code)  [NEU]
   - Greifer 1 Sekunde öffnen
   - Greifer schließen
4) INIT Pose anfahren (fest im Code)
5) LeRobot Recording starten
6) 5 Sekunden warten
7) wuerfe/throw_from_job.py mit --job ausführen + --result /tmp/throw_result.json
8) Landing-Zone per 1-Taste labeln (q/w/e/a/s/d/y/x/c, s=hit)
9) meta/throw_label.json schreiben (target + pos1/pos2 + release_progress + landing_zone + success + job_id + ts + run_name)
10) Warten bis LeRobot fertig ist (rec_proc.wait())
11) Danach automatisch patchen:
    - action[t] = state[t+1]
    - observation.task = target_xy_mm (aus job.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from xarm.wrapper import XArmAPI

import termios
import tty


# -----------------------------
# FESTE IP + INIT Pose
# -----------------------------
ROBOT_IP = "10.77.77.200"

INIT_JOINTS_RAD = [
    -1.570796,
     0.717557,
     1.192005,
     0.000000,
     0.474205,
    -1.570796,
]

# -----------------------------
# STILL Pose (NEU)
# -90, 45, 135, 0, -90, -90   (deg)
# -----------------------------
STILL_JOINTS_RAD = [
    -1.570796,  # -90
     0.785398,  # 45
     2.356194,  # 135
     0.000000,  # 0
    -1.570796,  # -90
    -1.570796,  # -90
]

# fixed RPY for top-down pick
FIX_R, FIX_P, FIX_YAW = -180, 0, 0


# ------------------------------------------------------------
# Landing zone keys (1-key labeling)
# ------------------------------------------------------------
ZONE_MAP = {
    "q": "top_left",
    "w": "top_center",
    "e": "top_right",
    "a": "mid_left",
    "s": "hit",
    "d": "mid_right",
    "y": "bot_left",
    "x": "bot_center",
    "c": "bot_right",
}
VALID_ZONE_KEYS = set(ZONE_MAP.keys())


def read_single_key(valid_keys=VALID_ZONE_KEYS) -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                continue
            ch = ch.lower()
            if ch in valid_keys:
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ------------------------------------------------------------
# Ball detection via subprocess (ball_finder.py)
# ------------------------------------------------------------
@dataclass
class BallDet:
    x_mm: float
    y_mm: float
    conf: float


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


# ------------------------------------------------------------
# Table -> Robot mapping (Affine)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Robot ops (pick stage only)
# ------------------------------------------------------------
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
) -> Tuple[bool, dict]:
    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[list]:
        code, joints = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code == 0 and joints is not None:
            return [float(v) for v in joints[:6]]

        # Roll wrap fallback (-180 <-> +180)
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

    meta = {"robot_pick_xy_mm": [rx, ry]}

    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False, meta
    code = arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move hover failed code={code}")
        return False, meta

    j_pick = ik_joints_for_pose_xyz_rpy_deg(rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False, meta
    code = arm.set_servo_angle(angle=j_pick, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move down failed code={code}")
        return False, meta

    time.sleep(0.25)

    arm.close_lite6_gripper(sync=True)

    time.sleep(0.3)

    j_lift = ik_joints_for_pose_xyz_rpy_deg(rx, ry, lift_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_lift is None:
        print("[ERR] IK lift failed")
        return False, meta
    code = arm.set_servo_angle(angle=j_lift, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] lift failed code={code}")
        return False, meta

    return True, meta


# ------------------------------------------------------------
# Job/Label helper
# ------------------------------------------------------------
def read_job_json(job_path: Path) -> dict:
    with open(job_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_job_json(job_path: Path, job: dict) -> None:
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)


def _safe_joint6(x) -> Optional[list]:
    if not isinstance(x, (list, tuple)) or len(x) != 6:
        return None
    try:
        return [float(v) for v in x]
    except Exception:
        return None


def write_throw_label_training_perfect(
    dataset_root: Path,
    *,
    run_name: str,
    job_id: int,
    target_x: Optional[float],
    target_y: Optional[float],
    job: dict,
    landing_zone: str,
    success: bool,
) -> None:
    pos1 = _safe_joint6(job.get("pos1", None))
    pos2 = _safe_joint6(job.get("pos2", None))
    if pos1 is None or pos2 is None:
        raise RuntimeError("Job JSON missing pos1/pos2 (each must be 6 joint angles in rad).")

    rp = job.get("release_progress", None)
    if rp is None:
        raise RuntimeError("Job JSON missing release_progress.")
    rp = float(rp)

    label = {
        "job_id": int(job_id),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "run_name": str(run_name),
        "target_xy_mm": [float(target_x), float(target_y)] if (target_x is not None and target_y is not None) else None,
        "pos1": pos1,
        "pos2": pos2,
        "release_progress": rp,
        "landing_zone": str(landing_zone),
        "success": bool(success),
    }

    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "throw_label.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2)


def run_auto_patch(dataset_root: Path) -> None:
    tools_dir = Path(__file__).resolve().parent
    patch_script = tools_dir / "patch_actions.py"

    cmd = [
        sys.executable,
        str(patch_script),
        "--dataset-root", str(dataset_root),
    ]

    print("[PATCH] Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[PATCH] Done. Dataset is now trainable & replay-safe.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, type=int, help="Job number N -> uses wuerfe/wurf_<N>.json")

    ap.add_argument("--x", type=float, default=None, help="Target X (mm)")
    ap.add_argument("--y", type=float, default=None, help="Target Y (mm)")

    ap.add_argument("--records-root", type=str, default="~/src/lite6xlerobot/records")
    ap.add_argument("--dataset-name", type=str, default="throws")
    ap.add_argument("--config-path", type=str, default="~/src/lite6xlerobot/configs/xarm_with_vitade.yaml")

    # detector (ball_finder)
    ap.add_argument("--cam", type=int, default=3)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ball-timeout-s", type=float, default=15.0)
    ap.add_argument("--verbose-detectors", action="store_true")

    # mapping + pick
    ap.add_argument("--table-to-robot-yaml", type=str, default="~/src/lite6xlerobot/calibrate/table_to_robot.yaml")
    ap.add_argument("--pick-y-offset-mm", type=float, default=0.0)
    ap.add_argument("--pick-x-offset-mm", type=float, default=0.0)
    ap.add_argument("--hover-z-mm", type=float, default=100.0)
    ap.add_argument("--pick-z-mm", type=float, default=-1.5)
    ap.add_argument("--lift-z-mm", type=float, default=120.0)
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # init pose motion (reused for STILL + INIT)
    ap.add_argument("--init-speed", type=float, default=1.0)
    ap.add_argument("--init-acc", type=float, default=1.0)

    # recorder timing
    ap.add_argument("--episode-time-s", type=float, default=12.0)

    args = ap.parse_args()

    tools_dir = Path(__file__).resolve().parent
    repo_root = tools_dir.parent

    job_path = repo_root / "wuerfe" / f"wurf_{int(args.job)}.json"
    if not job_path.exists():
        raise RuntimeError(f"Job JSON not found: {job_path}")

    throw_script = repo_root / "wuerfe" / "throw_from_job.py"
    if not throw_script.exists():
        raise RuntimeError(f"throw_from_job.py not found: {throw_script}")

    coord_dir = repo_root / "koordinaten"
    ball_script = coord_dir / "ball_finder.py"
    H_path = coord_dir / args.H
    if not ball_script.exists():
        raise RuntimeError(f"ball_finder.py not found: {ball_script}")
    if not H_path.exists():
        raise RuntimeError(f"H file not found: {H_path}")

    # -------------------------------------------------
    # TARGET XY handling
    # Priority:
    # 1) CLI args (override)
    # 2) job["target_xy_mm"]
    # -------------------------------------------------
    job = read_job_json(job_path)
    if args.x is not None and args.y is not None:
        target_x = float(args.x)
        target_y = float(args.y)
        job["target_xy_mm"] = [target_x, target_y]
        write_job_json(job_path, job)
        print(f"[JOB] Overridden target_xy_mm via CLI: {job['target_xy_mm']}")
    else:
        if "target_xy_mm" not in job:
            raise RuntimeError("Job JSON has no target_xy_mm and no --x/--y provided.")
        target_x, target_y = job["target_xy_mm"]
        print(f"[JOB] Using target_xy_mm from job: {job['target_xy_mm']}")

    records_root = Path(os.path.expanduser(args.records_root))
    dataset_parent = records_root / args.dataset_name
    dataset_parent.mkdir(parents=True, exist_ok=True)

    run_name = f"fs_{time.strftime('%Y-%m-%d_%H-%M-%S')}_job_wurf_{int(args.job)}"
    dataset_root = dataset_parent / run_name

    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    result_path = Path("/tmp") / f"throw_result_job_{int(args.job)}_{int(time.time())}.json"
    if result_path.exists():
        try:
            result_path.unlink()
        except Exception:
            pass

    arm = arm_connect()
    rec_proc: Optional[subprocess.Popen] = None

    try:
        # 1) FIND BALL
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

        # 2) PICK BALL
        mapper = TableToRobot(os.path.expanduser(args.table_to_robot_yaml))
        print("[PICK] Picking...")
        ok, _ = pick_ball_topdown(
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

        # 3) GO STILL POSE (NEU)
        print("[POSE] go_still_pose...")
        if not go_still_pose(arm, speed=args.init_speed, acc=args.init_acc, wait=True):
            raise RuntimeError("go_still_pose failed")
        print("[POSE] STILL reached")

        # STILL: gripper kurz auf/zu (MUSS blocking sein)
        print("[GRIP] open (still)...")
        arm.open_lite6_gripper(sync=True)
        time.sleep(3.0)

        print("[GRIP] close (still)...")
        arm.close_lite6_gripper(sync=True)
        time.sleep(0.3)

        # 4) GO INIT POSE
        print("[POSE] go_init_pose...")
        if not go_init_pose(arm, speed=args.init_speed, acc=args.init_acc, wait=True):
            raise RuntimeError("go_init_pose failed")
        print("[POSE] INIT reached")

        # 5) START RECORDING
        record_cmd = [
            "python", "-m", "lerobot.scripts.lerobot_record",
            "--config_path", os.path.expanduser(args.config_path),
            "--robot.ip", ROBOT_IP,

            "--teleop.type", "file_stream",
            "--teleop.path", "/tmp/lerobot_cmd.json",
            "--teleop.stale_s", "0.25",
            "--teleop.arm_action_keys", '["joint1","joint2","joint3","joint4","joint5","joint6"]',

            "--dataset.root", str(dataset_root),
            "--dataset.repo_id", f"local/{run_name}",
            "--dataset.single_task", f"throw_job_wurf_{int(args.job)}",
            "--dataset.push_to_hub", "false",
            "--dataset.num_episodes", "1",
            "--dataset.episode_time_s", str(args.episode_time_s),
            "--dataset.reset_time_s", "0",
            "--display_data", "false",
        ]

        print("[REC] Starting lerobot_record...")
        rec_proc = subprocess.Popen(record_cmd)

        # 6) fixed wait 5s
        time.sleep(5.0)

        # 7) throw
        print(f"[THROW] Running throw_from_job on {job_path.name} ...")
        subprocess.check_call([sys.executable, str(throw_script), "--job", str(job_path), "--result", str(result_path)])
        print("[THROW] Done.")

        # 8) landing_zone per key
        print("Landing-Zone: q w e / a s d / y x c   (s = HIT)")
        k = read_single_key()
        landing_zone = ZONE_MAP[k]
        success = (k == "s")
        print(f"[LABEL] landing_zone={landing_zone} success={success}")

        # 9) write label
        job_for_label = read_job_json(job_path)
        write_throw_label_training_perfect(
            dataset_root,
            run_name=run_name,
            job_id=int(args.job),
            target_x=target_x,
            target_y=target_y,
            job=job_for_label,
            landing_zone=landing_zone,
            success=success,
        )
        print("[LABEL] Wrote meta/throw_label.json")

        # 10) wait for recorder
        print("[REC] Waiting until lerobot_record finishes...")
        ret = rec_proc.wait()
        if ret != 0:
            raise RuntimeError(f"lerobot_record exited with code {ret}")
        print("[REC] Done.")

        # 11) Auto patch dataset
        print("[PATCH] Auto-patching dataset")
        run_auto_patch(dataset_root=dataset_root)

        print("[DONE] Recording + patch complete.")
        print(f"[DONE] Dataset ready at: {dataset_root}")

    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()