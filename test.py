#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time

# ================= MACHINE CONSTANTS =================
INIT_POSE = [-1.570796, 0.717557, 1.192005, 0.0, 0.474205, -1.570796]

SLOW_SPEED_DEG_S = 25.0
THROW_SPEED_DEG_S = 120.0
ACCEL_DEG_S2 = 350.0
COLLISION_SENS_THROW = 4
COLLISION_SENS_RESTORE = 3
MOVE_TIMEOUT_S = 10.0
SETTLE_S = 0.2

OPEN_TO0 = 1
OPEN_TO1 = 0

def deg2rad(x): return x * math.pi / 180.0

def load_job(path):
    with open(path, "r") as f: return json.load(f)

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

    def move_j(self, q, speed, accel, wait):
        self.arm.set_servo_angle(angle=q,
            speed=deg2rad(speed), mvacc=deg2rad(accel),
            is_radian=True, wait=wait, timeout=MOVE_TIMEOUT_S)

    def wait_stop(self):
        while self.arm.get_is_moving():
            time.sleep(0.01)

    def trigger(self, xyz, tol):
        self.arm.set_tgpio_digital_with_xyz(0, OPEN_TO0, xyz, tol)
        self.arm.set_tgpio_digital_with_xyz(1, OPEN_TO1, xyz, tol)

    def close_gripper(self):
        self.arm.close_lite6_gripper(True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    job = load_job(args.job)
    if job["release_xyz"] is None:
        raise RuntimeError("release_xyz missing → run calibration")

    robot = Lite6Robot(job["ip"])
    try:
        robot.connect()

        robot.move_j(INIT_POSE, SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True)
        robot.close_gripper()
        robot.move_j(job["pos1"], SLOW_SPEED_DEG_S, ACCEL_DEG_S2, True)
        time.sleep(SETTLE_S)

        robot.trigger(job["release_xyz"], job["xyz_tolerance_mm"])

        robot.move_j(job["pos2"], THROW_SPEED_DEG_S, ACCEL_DEG_S2, False)
        robot.wait_stop()

    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
