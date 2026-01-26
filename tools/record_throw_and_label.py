#!/usr/bin/env python3
"""record_throw_and_label.py

Ablauf (Episode):
1) Cup XY (Input fürs Modell) erfassen (CLI oder Prompt)
2) Ball finden (YOLO via koordinaten/ball_finder.py -> BALL_MM x y conf)
3) Ball aufnehmen (Top-Down Pick, IK, fixed RPY)
4) Zur INIT-Pose fahren (fixe Joint-Werte)
5) LeRobot Recording starten
6) Wurfskript ausführen
7) Recording endet (lerobot_record endet nach episode_time_s)
8) Label: target_xy_mm (getroffen) abfragen
9) Alles als JSONL loggen (Metadaten + Parameter + References)

Hinweis:
- Pick/Init sind NICHT Teil des LeRobot-Recordings (sauber für IL des Wurfs).
- Cup XY ist die "Task-Condition" fürs Modell: (cup_x, cup_y).

Änderungen (Jan 2026):
- WURF_PRESETS entfernt (throw_speed/throw_acc waren konstant/unnötig)
- release_at wird automatisch aus wurf_*.py gelesen (statisch per AST)
- optional: gripper re-init (enable toggle) ohne bewusstes open/close
"""

import argparse
import ast
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
# INIT Pose + Pick RPY
# -----------------------------
INIT_JOINTS_RAD = [
    -1.570796,  # J1  -90°
     0.717557,  # J2   41.1°
     1.192005,  # J3   68.3°
     0.000000,  # J4    0°
     0.474205,  # J5   27.2°
    -1.570796,  # J6  -90°
]

# Fixed TOP-DOWN orientation for picking (degrees)
FIX_R, FIX_P, FIX_YAW = -180, 0, 0


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
        raise RuntimeError("No parquet file found under data/")
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


def read_release_at_from_wurf(wurf_path: Path) -> Optional[float]:
    """
    Liest release_at aus wurf_*.py ohne das Skript auszuführen.

    Unterstützt:
    1) Top-level Variablen: RELEASE_AT, RELEASE_AT_PROGRESS, release_at
    2) WurfConfig(..., release_at=<zahl>, ...) als Keyword in einem Call
    """
    try:
        src = wurf_path.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        tree = ast.parse(src, filename=str(wurf_path))
    except SyntaxError:
        return None

    # 1) Fall: Top-level assignment
    wanted = {"RELEASE_AT", "RELEASE_AT_PROGRESS", "release_at"}
    found = {}

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                try:
                    val = ast.literal_eval(node.value)
                    if isinstance(val, (int, float)):
                        found[name] = float(val)
                except Exception:
                    pass

    for key in ["RELEASE_AT", "RELEASE_AT_PROGRESS", "release_at"]:
        if key in found:
            return found[key]

    # 2) Fall: WurfConfig(..., release_at=0.40, ...)
    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.value = None

        def visit_Call(self, node: ast.Call):
            # check function name: WurfConfig(...)
            fn_name = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr  # e.g. something.WurfConfig

            if fn_name == "WurfConfig":
                for kw in node.keywords:
                    if kw.arg == "release_at":
                        try:
                            v = ast.literal_eval(kw.value)
                            if isinstance(v, (int, float)):
                                self.value = float(v)
                                return  # stop early
                        except Exception:
                            pass

            # keep searching
            self.generic_visit(node)

    f = Finder()
    f.visit(tree)
    return f.value

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
) -> Tuple[bool, dict]:
    """
    Top-down pick: fixed RPY, only Z changes. Uses IK -> servo_angle.

    Returns (ok, meta) where meta includes robot-space pick XY used.
    """

    def ik_joints_for_pose_xyz_rpy_deg(x, y, z, r, p, yaw) -> Optional[list]:
        code, joints = arm.get_inverse_kinematics(
            [float(x), float(y), float(z), float(r), float(p), float(yaw)],
            input_is_radian=False,
            return_is_radian=True,
        )
        if code != 0 or joints is None:
            return None
        return [float(v) for v in joints[:6]]

    # open gripper before pick (einmal, nicht mehrfach)
    arm.open_lite6_gripper(sync=True)
    time.sleep(0.10)

    rx, ry = mapper.map(ball.x_mm, ball.y_mm)
    rx = rx + float(pick_x_offset_mm)
    ry = ry + float(pick_y_offset_mm)

    meta = {
        "robot_pick_xy_mm": [rx, ry],
    }

    # 1) hover
    j_hover = ik_joints_for_pose_xyz_rpy_deg(rx, ry, hover_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_hover is None:
        print("[ERR] IK hover failed")
        return False, meta
    code = arm.set_servo_angle(angle=j_hover, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move hover failed code={code}")
        return False, meta

    # 2) down
    j_pick = ik_joints_for_pose_xyz_rpy_deg(rx, ry, pick_z_mm, FIX_R, FIX_P, FIX_YAW)
    if j_pick is None:
        print("[ERR] IK pick failed")
        return False, meta
    code = arm.set_servo_angle(angle=j_pick, speed=ik_speed, mvacc=ik_acc, is_radian=True, wait=True)
    if code != 0:
        print(f"[ERR] move down failed code={code}")
        return False, meta

    # 3) close
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.10)

    # 4) lift
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
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wurf", type=int, required=True)

    ap.add_argument("--records-root", type=str, default="~/src/lite6xlerobot/records")
    ap.add_argument("--dataset-name", type=str, default="throws")
    ap.add_argument("--config-path", type=str, default="~/src/lite6xlerobot/configs/xarm_with_vitade.yaml")
    ap.add_argument("--robot-ip", type=str, default="10.77.77.200")

    # Cup input (Model-Condition)
    ap.add_argument("--cup-x-mm", type=float, default=None)
    ap.add_argument("--cup-y-mm", type=float, default=None)
    ap.add_argument("--ask-cup", action="store_true", help="Wenn gesetzt, fragt immer nach cup_x/cup_y.")

    # detector (ball_finder)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ball-timeout-s", type=float, default=15.0)
    ap.add_argument("--verbose-detectors", action="store_true")

    # mapping + pick
    ap.add_argument("--table-to-robot-yaml", type=str, default="~/src/lite6xlerobot/calibrate/table_to_robot.yaml")
    ap.add_argument("--pick-y-offset-mm", type=float, default=-27.0)
    ap.add_argument("--pick-x-offset-mm", type=float, default=3.7)
    ap.add_argument("--hover-z-mm", type=float, default=100.0)
    ap.add_argument("--pick-z-mm", type=float, default=15.5)
    ap.add_argument("--lift-z-mm", type=float, default=120.0)
    ap.add_argument("--ik-speed", type=float, default=1.0)
    ap.add_argument("--ik-acc", type=float, default=1.0)

    # init pose motion
    ap.add_argument("--init-speed", type=float, default=1.0)
    ap.add_argument("--init-acc", type=float, default=1.0)

    # recorder timing
    ap.add_argument("--pre-rec-wait-s", type=float, default=7.0, help="Wartezeit nach Start lerobot_record")
    ap.add_argument("--episode-time-s", type=float, default=15.0, help="dataset.episode_time_s")

    args = ap.parse_args()

    # --------------------------------------------------------
    # Resolve repo paths
    # --------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == "tools" else script_dir

    coord_dir = repo_root / "koordinaten"
    ball_script = coord_dir / "ball_finder.py"
    H_path = coord_dir / args.H

    wurf_path = repo_root / "wuerfe" / f"wurf_{args.wurf}.py"

    if not ball_script.exists():
        raise RuntimeError(f"ball_finder.py not found: {ball_script}")
    if not H_path.exists():
        raise RuntimeError(f"H file not found: {H_path}")
    if not wurf_path.exists():
        raise RuntimeError(f"Wurfskript nicht gefunden: {wurf_path}")

    # Read release_at from wurf script (statisch)
    release_at = read_release_at_from_wurf(wurf_path)
    if release_at is None:
        print(f"[WARN] Konnte release_at nicht aus {wurf_path.name} lesen. "
              f"Bitte im Wurfskript z.B. RELEASE_AT = 0.35 setzen.")

    # dataset dirs
    records_root = Path(os.path.expanduser(args.records_root))
    dataset_parent = records_root / args.dataset_name
    dataset_parent.mkdir(parents=True, exist_ok=True)

    run_name = f"fs_{time.strftime('%Y-%m-%d_%H-%M-%S')}_wurf{args.wurf:02d}"
    dataset_root = make_unique_dir(dataset_parent / run_name)

    # --------------------------------------------------------
    # Cup XY (Model input)
    # --------------------------------------------------------
    cup_x = args.cup_x_mm
    cup_y = args.cup_y_mm
    if args.ask_cup or cup_x is None or cup_y is None:
        print("[CUP] Bitte Cup-Koordinate (Input fürs Modell) eingeben (mm). 'skip' erlaubt.")
        cx = ask_float("cup_x_mm: ")
        cy = ask_float("cup_y_mm: ")
        if cx is not None and cy is not None:
            cup_x, cup_y = cx, cy

    print("\n=== RUN ===")
    print("run_name  :", run_name)
    print("wurf      :", wurf_path)
    print("dataset   :", dataset_root)
    if cup_x is not None and cup_y is not None:
        print(f"cup_xy    : ({cup_x:.1f},{cup_y:.1f}) mm")
    else:
        print("cup_xy    : None")
    print("release_at:", release_at)
    print("============\n")

    # --------------------------------------------------------
    # 0) CONNECT ROBOT (direct control)
    # --------------------------------------------------------
    print("[ROBOT] Connect...")
    arm = arm_connect(args.robot_ip)

    try:
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
        t_ball0 = time.time()
        ball = run_and_capture_first_ball(ball_cmd, timeout_s=args.ball_timeout_s, verbose=args.verbose_detectors)
        t_ball1 = time.time()

        if ball is None:
            raise RuntimeError("Ball not found (timeout)")

        print(f"[BALL] OK ({ball.x_mm:.1f},{ball.y_mm:.1f}) conf={ball.conf:.2f} (dt={t_ball1-t_ball0:.2f}s)")

        # --------------------------------------------------------
        # 2) PICK BALL
        # --------------------------------------------------------
        mapper = TableToRobot(os.path.expanduser(args.table_to_robot_yaml))
        print("[PICK] Picking (top-down)...")
        t_pick0 = time.time()
        ok, pick_meta = pick_ball_topdown(
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
        t_pick1 = time.time()
        if not ok:
            raise RuntimeError("Pick failed")
        print(f"[PICK] OK (dt={t_pick1-t_pick0:.2f}s)")

        # --------------------------------------------------------
        # 3) GO INIT POSE
        # --------------------------------------------------------
        print("[POSE] go_init_pose...")
        t_init0 = time.time()
        if not go_init_pose(arm, speed=args.init_speed, acc=args.init_acc, wait=True):
            raise RuntimeError("go_init_pose failed")
        t_init1 = time.time()
        print(f"[POSE] INIT reached (dt={t_init1-t_init0:.2f}s)")

        # --------------------------------------------------------
        # 4) START RECORDING (LeRobot)
        # --------------------------------------------------------
        record_cmd = [
            "python", "-m", "lerobot.scripts.lerobot_record",
            "--config_path", os.path.expanduser(args.config_path),
            "--robot.ip", args.robot_ip,

            # teleop: file_stream
            "--teleop.type", "file_stream",
            "--teleop.path", "/tmp/lerobot_cmd.json",
            "--teleop.stale_s", "0.25",
            "--teleop.arm_action_keys", '["joint1","joint2","joint3","joint4","joint5","joint6"]',

            # dataset
            "--dataset.root", str(dataset_root),
            "--dataset.repo_id", f"local/{run_name}",
            "--dataset.single_task", f"wurf_{args.wurf}",
            "--dataset.push_to_hub", "false",
            "--dataset.num_episodes", "1",
            "--dataset.episode_time_s", str(args.episode_time_s),
            "--dataset.reset_time_s", "0",
            "--display_data", "false",
        ]

        print("[REC] Starting lerobot_record...")
        t_rec0 = time.time()
        rec_proc = subprocess.Popen(record_cmd)
        time.sleep(max(0.0, float(args.pre_rec_wait_s)))

        # --------------------------------------------------------
        # 5) THROW (Wurfskript)
        # --------------------------------------------------------
        print("[THROW] Running wurf script...")
        t_throw0 = time.time()
        subprocess.check_call([sys.executable, str(wurf_path)])
        t_throw1 = time.time()
        print(f"[THROW] Done (dt={t_throw1-t_throw0:.2f}s)")

        # lerobot_record should stop by itself after episode_time_s
        rec_ret = rec_proc.wait()
        t_rec1 = time.time()
        if rec_ret != 0:
            raise RuntimeError(f"lerobot_record exited with code {rec_ret}")
        print(f"[REC] Done (total dt={t_rec1-t_rec0:.2f}s)")

        # --------------------------------------------------------
        # 6) LABEL (Target)
        # --------------------------------------------------------
        print("[LABEL] Eingabe (mm) – 'skip' wenn nicht gemessen.")
        tx = ask_float("target_x_mm: ")
        ty = ask_float("target_y_mm: ")

        chunk, file = find_latest_data_file(dataset_root)

        # --------------------------------------------------------
        # 7) LOG JSONL (everything we need)
        # --------------------------------------------------------
        label = {
            "ts": now_iso_local(),
            "run_name": run_name,
            "dataset_name": args.dataset_name,
            "dataset_root": str(dataset_root),
            "wurf_id": args.wurf,
            "wurf_path": str(wurf_path),

            # Data reference (join key)
            "data_ref": {"chunk": chunk, "file": file, "episode_index": 0},

            # Model condition / input
            "cup_xy_mm": [cup_x, cup_y] if cup_x is not None and cup_y is not None else None,

            # Label / outcome (optional)
            "target_xy_mm": [tx, ty] if tx is not None and ty is not None else None,

            # Ball detection (table space)
            "ball_xy_mm": [ball.x_mm, ball.y_mm],
            "ball_conf": ball.conf,

            # Pick & init metadata
            "init_joints_rad": INIT_JOINTS_RAD,
            "pick_meta": {
                "robot_pick_xy_mm": pick_meta.get("robot_pick_xy_mm"),
                "hover_z_mm": args.hover_z_mm,
                "pick_z_mm": args.pick_z_mm,
                "lift_z_mm": args.lift_z_mm,
                "pick_y_offset_mm": args.pick_y_offset_mm,
                "pick_x_offset_mm": args.pick_x_offset_mm,
                "topdown_rpy_deg": [FIX_R, FIX_P, FIX_YAW],
                "ik_speed": args.ik_speed,
                "ik_acc": args.ik_acc,
                "table_to_robot_yaml": os.path.expanduser(args.table_to_robot_yaml),
            },

            # Wurf parameters (nur das, was du wirklich brauchst)
            "throw_params": {
                "release_at": release_at,
                "episode_time_s": float(args.episode_time_s),
            },

            # Detector configuration used
            "detector": {
                "cam": int(args.cam),
                "H_path": str(H_path),
                "device": args.device,
                "ball_finder_cmd": ball_cmd,
                "ball_timeout_s": float(args.ball_timeout_s),
            },

            # Orchestration timings
            "timing_s": {
                "ball_find": round(t_ball1 - t_ball0, 4),
                "pick": round(t_pick1 - t_pick0, 4),
                "go_init": round(t_init1 - t_init0, 4),
                "throw": round(t_throw1 - t_throw0, 4),
                "record_total": round(t_rec1 - t_rec0, 4),
                "pre_rec_wait_s": float(args.pre_rec_wait_s),
            },
        }

        labels_path = dataset_parent / "labels.jsonl"
        append_jsonl(labels_path, label)
        print("\n[OK] Appended ->", labels_path)
        print("[OK] Done.\n")

    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
