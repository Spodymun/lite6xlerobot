#!/usr/bin/env python3
"""record_throw_and_label.py

Gewünschter Ablauf (Episode):

1) Ball finden (YOLO via koordinaten/ball_finder.py -> BALL_MM x y conf)
2) Ball aufnehmen (Top-Down Pick, IK, fixed RPY, nur Z fährt runter)
3) Zur INIT-Pose fahren (fixe Joint-Werte)
4) LeRobot Recording starten
5) Wurfskript ausführen
6) Recording endet (wenn lerobot_record fertig ist)
7) Optional: Label schreiben (target_xy_mm)

Hinweis:
- Dieses Skript nutzt dieselbe Pick-Logik wie run_autonomous.py, aber startet
  *LeRobot* Recording (lerobot_record) statt eines eigenen NPZ-Recorders.
- Pfade sind relativ zum Repo:
    koordinaten/ball_finder.py
    koordinaten/H.npy
    wuerfe/wurf_<n>.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from xarm.wrapper import XArmAPI


# -----------------------------
# FIX: INIT Pose + Pick RPY
# (aus run_autonomous.py)
# -----------------------------
INIT_JOINTS_RAD = [
    -1.5707963267948966,
     0.7173303225696694,
     1.1920598791121269,
    -0.10122909661567112,
     0.4886921905584123,
    -1.5707963267948966,
]
# Fixed TOP-DOWN orientation for picking (degrees)
FIX_R, FIX_P, FIX_YAW = -178.6, 2.1, 7.3


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def now_iso_local() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ask_float(prompt: str):
    s = input(prompt).strip()
    if not s or s.lower() in ("skip", "none", "-"):
        return None
    return float(s)


def newest_parquet_in(dirpath: Path) -> Optional[Path]:
    files = sorted(
        dirpath.glob("**/*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def find_latest_data_file(dataset_root: Path) -> Tuple[str, str]:
    data_root = dataset_root / "data"
    newest = newest_parquet_in(data_root)
    if newest is None:
        raise RuntimeError("No parquet file found")
    return newest.parent.name, newest.name


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        p = Path(str(path) + f"_v{i}")
        if not p.exists():
            return p
        i += 1


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
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        self.A = np.array(d["A"], dtype=float)
        self.b = np.array(d["b"], dtype=float)

    def map(self, x_table_mm: float, y_table_mm: float) -> Tuple[float, float]:
        p = np.array([x_table_mm, y_table_mm], dtype=float)
        r = self.A @ p + self.b
        return float(r[0]), float(r[1])


# ------------------------------------------------------------
# Robot ops
# ------------------------------------------------------------
def arm_connect(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    return arm


def go_home(arm: XArmAPI, *, wait: bool = True) -> bool:
    code = arm.move_gohome(wait=wait)
    if code != 0:
        print(f"[ERR] move_gohome failed code={code}")
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)
        return False
    return True


def go_init_pose(arm: XArmAPI, *, speed: float, acc: float, wait: bool = True) -> bool:
    code = arm.set_servo_angle(
        angle=INIT_JOINTS_RAD,
        speed=speed,
        mvacc=acc,
        is_radian=True,
        wait=wait,
    )
    if code != 0:
        print(f"[ERR] go_init_pose failed code={code}")
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
    """Top-down pick: fixed RPY, only Z changes. Uses IK -> servo_angle."""

    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[list]:
        code, joints = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code != 0 or joints is None:
            return None
        return [float(v) for v in joints[:6]]

    # open gripper before pick
    arm.open_lite6_gripper(sync=True)
    time.sleep(0.1)

    rx, ry = mapper.map(ball.x_mm, ball.y_mm)
    rx = rx + float(pick_x_offset_mm)
    ry = ry + float(pick_y_offset_mm)

    # 1) hover
    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False
    code = arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move hover failed code={code}")
        return False

    # 2) down
    j_pick = ik_joints_for_pose_xyz_rpy_deg(rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False
    code = arm.set_servo_angle(angle=j_pick, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move down failed code={code}")
        return False

    # 3) close
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.1)

    # 4) lift
    j_lift = ik_joints_for_pose_xyz_rpy_deg(rx, ry, lift_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_lift is None:
        print("[ERR] IK lift failed")
        return False
    code = arm.set_servo_angle(angle=j_lift, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] lift failed code={code}")
        return False

    return True


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wurf", type=int, required=True)

    ap.add_argument("--records-root", type=str, default="~/src/lite6xlerobot/records")
    ap.add_argument("--dataset-name", type=str, default="throws")
    ap.add_argument("--config-path", type=str, default="~/src/lite6xlerobot/configs/xarm_with_vitade.yaml")
    ap.add_argument("--robot-ip", type=str, default="10.77.77.200")

    # detector
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ball-timeout-s", type=float, default=15.0)
    ap.add_argument("--verbose-detectors", action="store_true")

    # mapping + pick
    ap.add_argument(
        "--table-to-robot-yaml",
        type=str,
        default="~/src/lite6xlerobot/calibrate/table_to_robot.yaml",
    )
    ap.add_argument("--pick-y-offset-mm", type=float, default=-30.0)
    ap.add_argument("--pick-x-offset-mm", type=float, default=3.0)
    ap.add_argument("--hover-z-mm", type=float, default=100.0)
    ap.add_argument("--pick-z-mm", type=float, default=15.5)
    ap.add_argument("--lift-z-mm", type=float, default=120.0)
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # init pose motion
    ap.add_argument("--init-speed", type=float, default=1.0)
    ap.add_argument("--init-acc", type=float, default=1.0)

    # home behavior
    ap.add_argument("--skip-home-start", action="store_true")

    # recorder timing
    ap.add_argument("--pre-rec-wait-s", type=float, default=7.0, help="Wartezeit nach Start lerobot_record")
    ap.add_argument("--episode-time-s", type=float, default=15.0, help="dataset.episode_time_s")

    args = ap.parse_args()

    # --------------------------------------------------------
    # Paths (relativ zum Script-Ordner)
    # --------------------------------------------------------
    script_dir = Path(__file__).resolve().parent

    # wenn wir in tools/ sind → parent ist repo root
    repo_root = script_dir.parent if script_dir.name == "tools" else script_dir

    # koordinaten
    coord_dir = repo_root / "koordinaten"

    ball_script = coord_dir / "ball_finder.py"
    H_path = coord_dir / args.H

    # wuerfe
    wurf_path = repo_root / "wuerfe" / f"wurf_{args.wurf}.py"

    if not ball_script.exists():
        raise RuntimeError(f"ball_finder.py not found: {ball_script}")
    if not H_path.exists():
        raise RuntimeError(f"H.npy not found: {H_path}")
    if not wurf_path.exists():
        raise RuntimeError(f"Wurfskript nicht gefunden: {wurf_path}")

    # dataset dirs
    records_root = Path(os.path.expanduser(args.records_root))
    dataset_parent = records_root / args.dataset_name
    dataset_parent.mkdir(parents=True, exist_ok=True)

    run_name = f"fs_{time.strftime('%Y-%m-%d_%H-%M-%S')}_wurf{args.wurf:02d}"
    dataset_root = make_unique_dir(dataset_parent / run_name)

    print("\n=== RUN ===")
    print("run:", run_name)
    print("wurf:", wurf_path)
    print("============\n")

    # --------------------------------------------------------
    # 0) CONNECT ROBOT
    # --------------------------------------------------------
    print("[ROBOT] Connect...")
    arm = arm_connect(args.robot_ip)

    try:
        # optional home at start
        arm.open_lite6_gripper(sync=True)

        # --------------------------------------------------------
        # 1) FIND BALL
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # 2) PICK BALL
        # --------------------------------------------------------
        mapper = TableToRobot(os.path.expanduser(args.table_to_robot_yaml))
        print("[PICK] Picking (top-down)...")
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

        # --------------------------------------------------------
        # 3) GO INIT POSE
        # --------------------------------------------------------
        print("[POSE] go_init_pose...")
        if not go_init_pose(arm, speed=args.init_speed, acc=args.init_acc, wait=True):
            raise RuntimeError("go_init_pose failed")

        # --------------------------------------------------------
        # 4) START RECORDING (LeRobot)
        # --------------------------------------------------------
        record_cmd = [
            "python", "-m", "lerobot.scripts.lerobot_record",
            "--config_path", os.path.expanduser(args.config_path),
            "--robot.ip", args.robot_ip,

            # teleop: file_stream (wie dein funktionierender Start)
            "--teleop.type", "file_stream",
            "--teleop.path", "/tmp/lerobot_cmd.json",
            "--teleop.stale_s", "0.25",
            "--teleop.arm_action_keys", '["joint1","joint2","joint3","joint4","joint5","joint6"]',
            "--teleop.gripper_action_key", "gripper.pos",

            # dataset
            "--dataset.root", str(dataset_root),
            "--dataset.repo_id", f"local/{run_name}",
            "--dataset.single_task", "file_stream test",  # oder f"wurf_{args.wurf}"
            "--dataset.push_to_hub", "false",
            "--dataset.num_episodes", "1",
            "--dataset.episode_time_s", str(args.episode_time_s),
            "--dataset.reset_time_s", "0",
            "--display_data", "false",
        ]

        print("[REC] Starting lerobot_record...")
        rec_proc = subprocess.Popen(record_cmd)
        time.sleep(max(0.0, float(args.pre_rec_wait_s)))

        # --------------------------------------------------------
        # 5) THROW (Wurfskript)
        # --------------------------------------------------------
        print("[THROW] Running wurf script...")
        subprocess.check_call([sys.executable, str(wurf_path)])

        # lerobot_record should stop by itself after episode_time_s
        rec_proc.wait()

        # --------------------------------------------------------
        # 6) LABEL
        # --------------------------------------------------------
        print("[LABEL] Eingabe")
        tx = ask_float("target_x_mm: ")
        ty = ask_float("target_y_mm: ")

        chunk, file = find_latest_data_file(dataset_root)

        label = {
            "ts": now_iso_local(),
            "run_name": run_name,
            "wurf_id": args.wurf,
            "data_ref": {"chunk": chunk, "file": file, "episode_index": 0},
            "target_xy_mm": [tx, ty] if tx is not None and ty is not None else None,
            "ball_xy_mm": [ball.x_mm, ball.y_mm],
            "ball_conf": ball.conf,
            "init_joints_rad": INIT_JOINTS_RAD,
            "pick_params": {
                "hover_z_mm": args.hover_z_mm,
                "pick_z_mm": args.pick_z_mm,
                "lift_z_mm": args.lift_z_mm,
                "pick_y_offset_mm": args.pick_y_offset_mm,
                "pick_x_offset_mm": args.pick_x_offset_mm,
                "topdown_rpy_deg": [FIX_R, FIX_P, FIX_YAW],
            },
        }

        append_jsonl(dataset_parent / "labels.jsonl", label)
        print("\n[OK] Done.")

    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()