#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from typing import Any, Callable, Dict, List, Tuple

# ================= MACHINE CONSTANTS =================
INIT_POSE = [-1.570796, 0.717557, 1.192005, 0.0, 0.474205, -1.570796]

SLOW_SPEED_DEG_S = 25.0
THROW_SPEED_DEG_S = 120.0
ACCEL_DEG_S2 = 350.0
COLLISION_SENS_THROW = 4
COLLISION_SENS_RESTORE = 3
MOVE_TIMEOUT_S = 10.0
SETTLE_S = 0.2

API_RETRIES = 3
API_BACKOFF_S = 0.08
AUTO_CLEAR = False

# Joint limits
LITE6_JOINT_LIMITS_DEG = {
    1: (-360.0, 360.0),
    2: (-150.0, 150.0),
    3: (-3.5, 300.0),
    4: (-360.0, 360.0),
    5: (-124.0, 124.0),
    6: (-360.0, 360.0),
}

def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi

def load_job(path):
    with open(path, "r") as f: return json.load(f)

def save_job(path, job):
    with open(path, "w") as f: json.dump(job, f, indent=2)

class Lite6Robot:
    def __init__(self, ip):
        self.ip = ip
        self.arm = None

    def connect(self):
        from xarm.wrapper import XArmAPI
        self.arm = XArmAPI(self.ip, is_radian=True, check_joint_limit=False)
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def disconnect(self):
        if self.arm: self.arm.disconnect()

    def move_j(self, q, speed, accel, wait, label):
        speed_rad, acc_rad = deg2rad(speed), deg2rad(accel)
        self.arm.set_servo_angle(angle=q, speed=speed_rad, mvacc=acc_rad,
                                 is_radian=True, wait=wait, timeout=MOVE_TIMEOUT_S)

    def get_tcp_pose(self):
        code, pose = self.arm.get_position(is_radian=False)
        if code != 0: raise RuntimeError("get_position failed")
        return pose[:3]

    @staticmethod
    def interp(q1, q2, t):
        return [q1[i] + t*(q2[i]-q1[i]) for i in range(6)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    job = load_job(args.job)
    robot = Lite6Robot(job["ip"])
    try:
        robot.connect()
        robot.move_j(INIT_POSE, SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "init")
        robot.move_j(job["pos1"], SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "pos1")
        time.sleep(SETTLE_S)

        mid = robot.interp(job["pos1"], job["pos2"], job["release_progress"])
        robot.move_j(mid, SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "mid")
        time.sleep(SETTLE_S)

        job["release_xyz"] = robot.get_tcp_pose()

        robot.move_j(job["pos1"], SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True, "back")

        save_job(args.job, job)
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
