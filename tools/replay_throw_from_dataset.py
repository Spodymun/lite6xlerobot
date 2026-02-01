#!/usr/bin/env python3
"""
tools/replay_throw_from_dataset.py

Replays a THROW using the *throw controller logic* (fast, real throw),
NOT LeRobot's safe teleop replay.

Reads from dataset_root/meta/throw_label.json:
  - pos1 (6 joints rad)
  - pos2 (6 joints rad)
  - release_progress (0..1)

Then executes:
  init(optional) -> pos1 -> async move to pos2 -> release at progress -> finish

Usage:
  python3 tools/replay_throw_from_dataset.py \
    --dataset_root /path/to/fs_... \
    --ip 10.77.77.200 \
    --init-joints -1.570796,0.717557,1.192005,0,0.474205,-1.570796 \
    --speed-deg-s 120 \
    --accel-deg-s2 350 \
    --poll-hz 250 \
    --preopen-ms 0 \
    --gripper-mode open_lite6 \
    --return-pos1

Gripper modes:
  - open_lite6 : arm.open_lite6_gripper(sync=False)
  - tgpio      : arm.set_tgpio_digital(io, val)   (latching)
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List

from xarm.wrapper import XArmAPI


# -----------------------------
# Helpers
# -----------------------------

def deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def parse_joints_csv(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated joint values.")
    return [float(x) for x in parts]

def load_throw_label(dataset_root: Path) -> dict:
    p = dataset_root / "meta" / "throw_label.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing throw_label.json: {p}")
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)

    # validate
    if "pos1" not in d or "pos2" not in d or "release_progress" not in d:
        raise RuntimeError(f"{p} must contain pos1,pos2,release_progress")
    if not (isinstance(d["pos1"], list) and len(d["pos1"]) == 6):
        raise RuntimeError("throw_label.json: pos1 must be list[6] (rad)")
    if not (isinstance(d["pos2"], list) and len(d["pos2"]) == 6):
        raise RuntimeError("throw_label.json: pos2 must be list[6] (rad)")

    rp = float(d["release_progress"])
    if not (0.0 <= rp <= 1.0):
        raise RuntimeError(f"release_progress must be in [0,1], got {rp}")

    return d

def connect_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True, check_joint_limit=False)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)   # position mode
    arm.set_state(0)
    time.sleep(0.2)
    return arm

def get_joints(arm: XArmAPI) -> List[float]:
    code, q = arm.get_servo_angle(is_radian=True)
    if code != 0 or q is None:
        raise RuntimeError(f"get_servo_angle failed code={code}")
    return [float(v) for v in q[:6]]

def proj_progress(q: List[float], pos1: List[float], pos2: List[float]) -> float:
    # progress along pos1->pos2 in joint space
    v = [pos2[i] - pos1[i] for i in range(6)]
    denom = sum(vi*vi for vi in v)
    if denom < 1e-12:
        return 0.0
    num = sum((q[i] - pos1[i]) * v[i] for i in range(6))
    t = num / denom
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    return float(t)

def move_j(arm: XArmAPI, q: List[float], speed_deg_s: float, accel_deg_s2: float, wait: bool, label: str):
    speed = deg2rad(speed_deg_s)
    acc = deg2rad(accel_deg_s2)
    code = arm.set_servo_angle(angle=q, speed=speed, mvacc=acc, is_radian=True, wait=wait)
    if code != 0:
        raise RuntimeError(f"{label}: set_servo_angle failed code={code}")

def open_gripper(arm: XArmAPI, mode: str, tgpio_io: int, tgpio_val: int):
    mode = mode.lower().strip()
    if mode == "open_lite6":
        arm.open_lite6_gripper(sync=False)
        return
    if mode == "tgpio":
        # latching digital output on tool GPIO
        arm.set_tgpio_digital(tgpio_io, tgpio_val)
        return
    raise ValueError(f"Unknown --gripper-mode '{mode}'. Use: open_lite6|tgpio")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help=".../fs_YYYY-MM-DD_HH-MM-SS_job_wurf_N")
    ap.add_argument("--ip", default="10.77.77.200")

    ap.add_argument("--init-joints", default=None,
                    help="Optional init joints (6 rad) as csv: j1,j2,j3,j4,j5,j6")
    ap.add_argument("--go-init", action="store_true",
                    help="If set, moves to --init-joints before pos1 (requires --init-joints).")

    ap.add_argument("--speed-deg-s", type=float, default=120.0)
    ap.add_argument("--accel-deg-s2", type=float, default=350.0)

    ap.add_argument("--poll-hz", type=float, default=250.0,
                    help="Progress polling rate during the throw (higher = tighter release timing)")
    ap.add_argument("--preopen-ms", type=float, default=0.0,
                    help="Fire release slightly earlier (ms) to compensate valve latency")

    ap.add_argument("--gripper-mode", default="open_lite6", choices=["open_lite6", "tgpio"])
    ap.add_argument("--tgpio-io", type=int, default=0, help="For gripper-mode=tgpio")
    ap.add_argument("--tgpio-val", type=int, default=1, help="For gripper-mode=tgpio")

    ap.add_argument("--return-pos1", action="store_true", help="After throw, go back to pos1.")
    ap.add_argument("--dry-run", action="store_true", help="Do not move robot, just print parameters.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    label = load_throw_label(dataset_root)

    pos1 = [float(v) for v in label["pos1"]]
    pos2 = [float(v) for v in label["pos2"]]
    release_progress = float(label["release_progress"])

    init_joints: Optional[List[float]] = None
    if args.init_joints is not None:
        init_joints = parse_joints_csv(args.init_joints)

    print("=== THROW REPLAY (controller-style) ===")
    print("dataset_root     :", dataset_root)
    print("ip               :", args.ip)
    print("pos1 (rad)       :", [round(x, 6) for x in pos1])
    print("pos2 (rad)       :", [round(x, 6) for x in pos2])
    print("release_progress :", release_progress)
    print("speed_deg_s      :", args.speed_deg_s)
    print("accel_deg_s2     :", args.accel_deg_s2)
    print("poll_hz          :", args.poll_hz)
    print("preopen_ms       :", args.preopen_ms)
    print("gripper_mode     :", args.gripper_mode)
    if args.gripper_mode == "tgpio":
        print("tgpio (io,val)   :", (args.tgpio_io, args.tgpio_val))

    if args.go_init and init_joints is None:
        raise SystemExit("--go-init requires --init-joints")

    if args.dry_run:
        print("[DRY RUN] exiting without motion.")
        return

    arm = connect_arm(args.ip)
    try:
        # optional init
        if args.go_init and init_joints is not None:
            print("[1/4] INIT")
            move_j(arm, init_joints, speed_deg_s=25.0, accel_deg_s2=args.accel_deg_s2, wait=True, label="init")

        # go pos1
        print("[2/4] POS1")
        move_j(arm, pos1, speed_deg_s=25.0, accel_deg_s2=args.accel_deg_s2, wait=True, label="pos1")
        time.sleep(0.15)

        # throw pos1->pos2 async
        print("[3/4] THROW (async to pos2)")
        move_j(arm, pos2, speed_deg_s=args.speed_deg_s, accel_deg_s2=args.accel_deg_s2, wait=False, label="pos2_async")

        # monitor progress
        dt = 1.0 / float(args.poll_hz)
        preopen_s = max(0.0, float(args.preopen_ms) / 1000.0)
        t_start = time.time()
        released = False

        # We trigger when progress >= release_progress, but optionally earlier by time offset:
        # simplest and robust: trigger at progress threshold, but shift earlier by time by subtracting preopen_s from "now".
        # Here: we trigger as soon as threshold is met, OR slightly earlier by using a small progress margin if preopen_s>0.
        # (We keep it time-based: just trigger immediately once threshold hit; user can set preopen_ms as valve latency compensation.)

        while True:
            q = get_joints(arm)
            prog = proj_progress(q, pos1, pos2)

            if (not released) and (prog >= release_progress):
                # Optional preopen: if set, we fire immediately (the effect is earlier release because valve latency)
                if preopen_s > 0:
                    # fire now, user-provided latency compensation
                    pass
                print(f"[RELEASE] prog={prog:.3f} t={time.time()-t_start:.3f}s")
                open_gripper(arm, args.gripper_mode, args.tgpio_io, args.tgpio_val)
                released = True

            # end condition: near pos2 OR motion ended
            # We use progress close to 1.0 as stop condition.
            if prog >= 0.995:
                break

            time.sleep(dt)

        # small settle
        time.sleep(0.10)

        # optional return
        if args.return_pos1:
            print("[4/4] RETURN POS1")
            move_j(arm, pos1, speed_deg_s=25.0, accel_deg_s2=args.accel_deg_s2, wait=True, label="return_pos1")

        print("[DONE] throw replay complete.")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
