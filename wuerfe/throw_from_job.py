#!/usr/bin/env python3
"""
wuerfe/throw_from_job.py

Reads a job JSON and executes the throw.
Additionally logs a robust release_xyz_actual by TCP entering the release sphere:
  dist(TCP_xyz, release_xyz_cmd) <= xyz_tolerance_mm

Writes a result JSON if --result is provided.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

from xarm.wrapper import XArmAPI

# Fixed INIT pose (always the same)
INIT_JOINTS_RAD = [
    -1.570796,
     0.717557,
     1.192005,
     0.000000,
     0.474205,
    -1.570796,
]

# Minimal constants (your "always same setup")
SLOW_SPEED_DEG_S = 25.0
THROW_SPEED_DEG_S = 500.0
ACCEL_DEG_S2 = 350.0
MOVE_TIMEOUT_S = 10.0
SETTLE_S = 0.2

COLL_SENS_THROW = 4
COLL_SENS_RESTORE = 3

OPEN_TO0 = 1
OPEN_TO1 = 0


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def safe_xyz(x) -> Optional[list]:
    if not isinstance(x, (list, tuple)) or len(x) < 3:
        return None
    return [float(x[0]), float(x[1]), float(x[2])]


def arm_connect(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True, check_joint_limit=False)
    arm.connect()
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)
    return arm


def move_j(arm: XArmAPI, joints_rad, speed_deg_s: float, accel_deg_s2: float, wait: bool, label: str):
    code = arm.set_servo_angle(
        angle=joints_rad,
        speed=deg2rad(speed_deg_s),
        mvacc=deg2rad(accel_deg_s2),
        is_radian=True,
        wait=wait,
        timeout=MOVE_TIMEOUT_S,
    )
    if code != 0:
        raise RuntimeError(f"move_j failed ({label}) code={code}")


def get_tcp_xyz(arm: XArmAPI) -> Optional[list]:
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        return None
    return [float(pose[0]), float(pose[1]), float(pose[2])]


def dist(a, b) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def watch_release_sphere_during_motion(arm: XArmAPI, release_xyz_cmd: list, tol_mm: float, timeout_s: float = 6.0, poll_hz: float = 200.0):
    """
    Poll TCP while motion runs; return (release_xyz_actual, min_dist_mm, samples).
    """
    dt = 1.0 / max(1.0, poll_hz)
    t0 = time.time()
    best = None
    release_actual = None
    samples = 0

    while True:
        if time.time() - t0 > timeout_s:
            return release_actual, best, samples

        xyz = get_tcp_xyz(arm)
        if xyz is not None:
            samples += 1
            d = dist(xyz, release_xyz_cmd)
            best = d if (best is None or d < best) else best
            if d <= tol_mm and release_actual is None:
                release_actual = xyz
                # don't return immediately; but usually it's fine to return right away
                return release_actual, best, samples

        # stop early if robot stopped moving and we already have best
        try:
            if not bool(arm.get_is_moving()) and best is not None:
                return release_actual, best, samples
        except Exception:
            pass

        time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to job json (e.g. wuerfe/wurf_1.json)")
    ap.add_argument("--result", default=None, help="Write result json here (optional)")
    args = ap.parse_args()

    job_path = Path(args.job)
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    ip = job.get("ip", "10.77.77.200")
    pos1 = job.get("pos1", None)
    pos2 = job.get("pos2", None)
    release_xyz_cmd = safe_xyz(job.get("release_xyz", None))
    tol = float(job.get("xyz_tolerance_mm", 10.0))

    if pos1 is None or pos2 is None:
        raise RuntimeError("Job json must contain pos1 and pos2 (joint rad lists).")
    if release_xyz_cmd is None:
        raise RuntimeError("Job json must contain release_xyz (TCP xyz in mm).")

    arm = arm_connect(ip)
    result = {
        "job_file": str(job_path),
        "ip": ip,
        "release_xyz_cmd": release_xyz_cmd,
        "xyz_tolerance_mm": tol,
        "release_xyz_actual": None,
        "min_dist_to_release_cmd_mm": None,
        "tcp_samples": 0,
        "ok": False,
        "error": None,
    }

    try:
        # Init pose
        move_j(arm, INIT_JOINTS_RAD, SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "init")
        time.sleep(SETTLE_S)

        # Prep
        arm.close_lite6_gripper(sync=False)
        time.sleep(1.0)

        # Go pos1 slow
        move_j(arm, pos1, SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "pos1")
        time.sleep(SETTLE_S)

        # Collision sens for throw
        arm.set_collision_sensitivity(COLL_SENS_THROW)

        # Arm-side position trigger (the actual release mechanism)
        arm.set_tgpio_digital_with_xyz(0, OPEN_TO0, release_xyz_cmd, tol)
        arm.set_tgpio_digital_with_xyz(1, OPEN_TO1, release_xyz_cmd, tol)

        # Throw pos1 -> pos2 non-blocking
        move_j(arm, pos2, THROW_SPEED_DEG_S, ACCEL_DEG_S2, False, "pos2_throw")

        # Watch during motion using SAME connection
        release_actual, best, samples = watch_release_sphere_during_motion(
            arm, release_xyz_cmd, tol, timeout_s=6.0, poll_hz=200.0
        )
        result["release_xyz_actual"] = release_actual
        result["min_dist_to_release_cmd_mm"] = best
        result["tcp_samples"] = samples

        # Wait for stop
        t0 = time.time()
        while bool(arm.get_is_moving()) and (time.time() - t0) < MOVE_TIMEOUT_S:
            time.sleep(0.01)

        # Restore
        arm.set_collision_sensitivity(COLL_SENS_RESTORE)

        result["ok"] = True

    except Exception as e:
        result["error"] = str(e)
        result["ok"] = False
        raise
    finally:
        try:
            if args.result:
                Path(args.result).parent.mkdir(parents=True, exist_ok=True)
                with open(args.result, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()