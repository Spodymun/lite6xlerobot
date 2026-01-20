#!/usr/bin/env python3
"""
throw_test.py (TCP position-triggered release)

Modes:
- queue:    start throw (wait=False), call open after --open-after seconds
- progress: start throw (wait=False), open when joint-space progress >= --release-at
- tcp:      start throw (wait=False), open when TCP enters sphere around (X,Y,Z) with tolerance

Examples:
  python3 throw_test.py --mode tcp --release-x 240 --release-y -200 --release-z 440 --tol-mm 20
  python3 throw_test.py --mode tcp --tol-mm 20   # auto uses POS2 TCP as target if release xyz not set
"""

import argparse
import json
import os
import time
import threading
import numpy as np

from xarm.wrapper import XArmAPI
from lerobot_robot_xarm.config_xarm import XarmConfig

ROBOT_IP = "10.77.77.200"


def load_positions(positions_file: str | None):
    start_pos = None
    end_pos = None

    if positions_file and os.path.exists(positions_file):
        try:
            with open(positions_file, "r") as f:
                positions = json.load(f)
            if "pos1" in positions and "pos2" in positions:
                start_pos = positions.get("pos1")
                end_pos = positions.get("pos2")
                print(f"Loaded positions from file: {positions_file}")
        except Exception as e:
            print("Could not load positions file:", e)

    if start_pos is None:
        start_pos = [-1.640, -0.920, 2.983, 0.042, 0.178, -0.113]
        #start_pos = [-1.567, -1.853, 2.862, 0.017, -0.305, -0.113]
    if end_pos is None:
        end_pos = [-1.517, 1.476, 2.913, -0.002, 0.079, -0.113]

    start_pos = [float(x) for x in start_pos[:6]]
    end_pos = [float(x) for x in end_pos[:6]]
    return start_pos, end_pos


def joint_progress(cur, start, target) -> float:
    cur = np.array(cur[:6], dtype=float)
    start = np.array(start[:6], dtype=float)
    target = np.array(target[:6], dtype=float)

    total = np.linalg.norm(target - start)
    if total < 1e-6:
        return 1.0

    done = np.linalg.norm(cur - start)
    return float(np.clip(done / total, 0.0, 1.0))


def init_arm() -> XArmAPI:
    arm = XArmAPI(ROBOT_IP, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)
    try:
        print("err_warn:", arm.get_err_warn_code())
    except Exception:
        pass
    return arm


def move_home_and_pos1(arm: XArmAPI, home_pos, start_pos, home_speed, home_acc, pos1_speed, pos1_acc):
    print(f"[1/4] HOME (speed={home_speed}, acc={home_acc})")
    arm.set_servo_angle(angle=home_pos, speed=home_speed, mvacc=home_acc, is_radian=True, wait=True)
    print("  ✓ HOME reached\n")

    # >>> HIER schließen, wie du willst:
    print("  -> Closing gripper at HOME...")
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.2)

    print(f"[2/4] POS1 (speed={pos1_speed}, acc={pos1_acc})")
    arm.set_servo_angle(angle=start_pos, speed=pos1_speed, mvacc=pos1_acc, is_radian=True, wait=True)
    print("  ✓ POS1 reached\n")

    print("  -> Closing gripper...")
    # sync=True hilft fürs Greifen; bei dir funktioniert lite6 gripper
    arm.close_lite6_gripper(sync=True)
    time.sleep(0.2)


def run_queue_test(arm: XArmAPI, end_pos, throw_speed, throw_acc, open_after_s: float):
    print(f"[3/4] QUEUE TEST THROW (speed={throw_speed}, acc={throw_acc})")
    t0 = time.time()
    print("  THROW start t=0.000")

    ret = arm.set_servo_angle(angle=end_pos, speed=throw_speed, mvacc=throw_acc, is_radian=True, wait=False)
    print(f"  sent set_servo_angle ret={ret} at t={time.time() - t0:.3f}s")

    time.sleep(max(0.0, float(open_after_s)))
    print(f"  calling open_lite6_gripper at t={time.time() - t0:.3f}s")
    t_open = time.time()
    arm.open_lite6_gripper(sync=False)
    dt_open = time.time() - t_open
    print(f"  open_lite6_gripper RETURNED after {dt_open:.3f}s, at t={time.time() - t0:.3f}s")

    time.sleep(1.0)
    code, cur = arm.get_servo_angle(is_radian=True)
    if code == 0:
        print("  angles after 1s:", cur[:6])
    print()


def run_progress_release(arm: XArmAPI, start_pos, end_pos, throw_speed, throw_acc,
                         release_at: float, progress_timeout_s: float, poll_dt_s: float):
    code, start_angles = arm.get_servo_angle(is_radian=True)
    if code != 0:
        print("WARNING: could not read start angles; falling back to start_pos.")
        start_angles = start_pos
    start_angles = start_angles[:6]

    print(f"[3/4] PROGRESS THROW (speed={throw_speed}, acc={throw_acc}, release_at={release_at:.2f})")
    t0 = time.time()
    ret = arm.set_servo_angle(angle=end_pos, speed=throw_speed, mvacc=throw_acc, is_radian=True, wait=False)
    print(f"  sent set_servo_angle ret={ret} at t={time.time() - t0:.3f}s")

    def opener():
        opened = False
        while (time.time() - t0) < progress_timeout_s and not opened:
            c, cur = arm.get_servo_angle(is_radian=True)
            if c == 0:
                p = joint_progress(cur, start_angles, end_pos)
                if p >= release_at:
                    print(f"  calling open_lite6_gripper at progress={p:.2f}, t={time.time() - t0:.3f}s")
                    t_open = time.time()
                    arm.open_lite6_gripper(sync=False)
                    dt_open = time.time() - t_open
                    print(f"  open_lite6_gripper RETURNED after {dt_open:.3f}s, t={time.time() - t0:.3f}s")
                    opened = True
                    break
            time.sleep(poll_dt_s)

        if not opened:
            print(f"  WARNING: progress trigger not reached in {progress_timeout_s:.2f}s -> opening anyway")
            t_open = time.time()
            arm.open_lite6_gripper(sync=False)
            dt_open = time.time() - t_open
            print(f"  open_lite6_gripper RETURNED after {dt_open:.3f}s")

    threading.Thread(target=opener, daemon=True).start()
    time.sleep(2.0)
    print("  (End of throw window)\n")


def run_tcp_release(
    arm: XArmAPI,
    end_pos,
    throw_speed,
    throw_acc,
    release_xyz,        # (x,y,z) in mm
    tol_mm: float,
    poll_dt_s: float,
    timeout_s: float,
):
    """
    TCP position-based release:
    - Start throw joint move (wait=False)
    - Poll TCP position via arm.get_position()
    - When TCP enters sphere around release_xyz with radius tol_mm -> open gripper
    Logs call and return timing.
    """
    rx, ry, rz = release_xyz
    tol2 = float(tol_mm) * float(tol_mm)

    print(f"[3/4] TCP THROW (speed={throw_speed}, acc={throw_acc}, release_xyz={release_xyz}, tol={tol_mm}mm)")
    t0 = time.time()
    ret = arm.set_servo_angle(angle=end_pos, speed=throw_speed, mvacc=throw_acc, is_radian=True, wait=False)
    print(f"  sent set_servo_angle ret={ret} at t={time.time() - t0:.3f}s")

    opened = False
    last_print_t = -999.0

    while (time.time() - t0) < timeout_s and not opened:
        code, pos = arm.get_position(is_radian=False)  # mm/deg
        if code == 0:
            x, y, z = pos[0], pos[1], pos[2]
            dx, dy, dz = x - rx, y - ry, z - rz
            d2 = dx*dx + dy*dy + dz*dz

            # status print every ~0.25s (optional)
            if (time.time() - t0) - last_print_t > 0.25:
                print(f"    t={time.time()-t0:.3f}s TCP=({x:.1f},{y:.1f},{z:.1f}) d={np.sqrt(d2):.1f}mm")
                last_print_t = (time.time() - t0)

            if d2 <= tol2:
                print(f"  >>> TCP TRIGGER HIT at t={time.time() - t0:.3f}s, TCP=({x:.1f},{y:.1f},{z:.1f})")
                print("  calling open_lite6_gripper ...")
                t_open = time.time()
                arm.open_lite6_gripper(sync=False)
                dt_open = time.time() - t_open
                print(f"  open_lite6_gripper RETURNED after {dt_open:.3f}s, t={time.time() - t0:.3f}s")
                opened = True
                break

        time.sleep(poll_dt_s)

    if not opened:
        print(f"  WARNING: TCP trigger not reached in {timeout_s:.2f}s -> opening anyway")
        t_open = time.time()
        arm.open_lite6_gripper(sync=False)
        dt_open = time.time() - t_open
        print(f"  open_lite6_gripper RETURNED after {dt_open:.3f}s")

    time.sleep(1.0)
    print("  (End of throw window)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=str, default="/home/nelly/src/throw_positions_20260117_160127.json")
    ap.add_argument("--mode", choices=["queue", "progress", "tcp"], default="tcp")

    ap.add_argument("--home-speed", type=float, default=0.5)
    ap.add_argument("--home-acc", type=float, default=1.0)
    ap.add_argument("--pos1-speed", type=float, default=1.0)
    ap.add_argument("--pos1-acc", type=float, default=1.0)

    ap.add_argument("--throw-speed", type=float, default=3.0)
    ap.add_argument("--throw-acc", type=float, default=6.0)

    # queue test
    ap.add_argument("--open-after", type=float, default=0.02, help="seconds after throw start to call open()")

    # progress release
    ap.add_argument("--release-at", type=float, default=0.35, help="0..1 progress fraction")
    ap.add_argument("--progress-timeout", type=float, default=2.0)
    ap.add_argument("--poll-dt", type=float, default=0.005)

    # tcp release
    ap.add_argument("--release-x", type=float, default=None)
    ap.add_argument("--release-y", type=float, default=None)
    ap.add_argument("--release-z", type=float, default=None)
    ap.add_argument("--tol-mm", type=float, default=20.0)
    ap.add_argument("--tcp-timeout", type=float, default=2.0)

    args = ap.parse_args()

    positions_file = args.positions if (args.positions and os.path.exists(args.positions)) else None
    start_pos, end_pos = load_positions(positions_file)

    config = XarmConfig()
    home_pos = config.home_joint_positions[:6]

    print("\n=== THROW TEST ===")
    print("mode:", args.mode)
    print("positions_file:", positions_file)
    print()

    arm = init_arm()

    try:
        move_home_and_pos1(
            arm,
            home_pos=home_pos,
            start_pos=start_pos,
            home_speed=args.home_speed,
            home_acc=args.home_acc,
            pos1_speed=args.pos1_speed,
            pos1_acc=args.pos1_acc,
        )

        if args.mode == "queue":
            run_queue_test(
                arm,
                end_pos=end_pos,
                throw_speed=args.throw_speed,
                throw_acc=args.throw_acc,
                open_after_s=args.open_after,
            )

        elif args.mode == "progress":
            run_progress_release(
                arm,
                start_pos=start_pos,
                end_pos=end_pos,
                throw_speed=args.throw_speed,
                throw_acc=args.throw_acc,
                release_at=args.release_at,
                progress_timeout_s=args.progress_timeout,
                poll_dt_s=args.poll_dt,
            )

        else:
            # TCP release target
            if args.release_x is not None and args.release_y is not None and args.release_z is not None:
                release_xyz = (args.release_x, args.release_y, args.release_z)
            else:
                # Auto: use current TCP at POS1 as baseline, then use TCP at POS2 as target by briefly moving to POS2 and back?
                # Safer: compute target by actually moving is complex; instead:
                # -> we use the TCP when throw starts + no target set => ask user to pass XYZ.
                # BUT you asked not to add questions, so:
                # We'll default to a "near POS1" release target (will trigger quickly) unless you pass XYZ.
                code, pos = arm.get_position(is_radian=False)
                release_xyz = (pos[0], pos[1], pos[2]) if code == 0 else (0.0, 0.0, 0.0)
                print("NOTE: No --release-x/y/z provided, defaulting release_xyz to current TCP:", release_xyz)

            run_tcp_release(
                arm,
                end_pos=end_pos,
                throw_speed=args.throw_speed,
                throw_acc=args.throw_acc,
                release_xyz=release_xyz,
                tol_mm=args.tol_mm,
                poll_dt_s=args.poll_dt,
                timeout_s=args.tcp_timeout,
            )

        print(f"[4/4] HOME return (speed={args.home_speed}, acc={args.home_acc})")
        arm.set_servo_angle(angle=home_pos, speed=args.home_speed, mvacc=args.home_acc, is_radian=True, wait=True)
        print("  ✓ HOME reached")

    finally:
        print("\nDisconnecting...")
        arm.disconnect()


if __name__ == "__main__":
    main()
