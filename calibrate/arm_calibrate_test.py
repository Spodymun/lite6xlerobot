#!/usr/bin/env python3
"""
move_to_table_xy.py

Fährt den xArm zu einer Position, die in TISCH-Koordinaten (mm) angegeben ist.

Pipeline:
  (x_table, y_table) [mm]  ->  (X_robot, Y_robot) [mm] via table_to_robot.yaml
  -> arm.set_position(X_robot, Y_robot, Z_const)

Run:
  python3 move_to_table_xy.py --ip 10.77.77.200 --calib table_to_robot.yaml --xt 360 --yt 800
"""

import time
import argparse
import numpy as np
import yaml
from xarm.wrapper import XArmAPI


def load_table_to_robot(yaml_path: str):
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    A = np.array(d["A"], dtype=float)
    b = np.array(d["b"], dtype=float).reshape(2,)
    if A.shape != (2, 2) or b.shape != (2,):
        raise ValueError(f"Bad calib shapes: A={A.shape}, b={b.shape}")
    return A, b


def map_table_to_robot(A: np.ndarray, b: np.ndarray, xt: float, yt: float):
    p = np.array([xt, yt], dtype=float)
    r = A @ p + b
    return float(r[0]), float(r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, required=True, help="Robot IP, z.B. 10.77.77.200")
    ap.add_argument("--calib", type=str, default="table_to_robot.yaml", help="table_to_robot.yaml")
    ap.add_argument("--xt", type=float, required=True, help="Tisch X (mm)")
    ap.add_argument("--yt", type=float, required=True, help="Tisch Y (mm)")
    ap.add_argument("--z", type=float, default=30.0, help="Konstante Z-Höhe (mm), Default 30")
    ap.add_argument("--speed", type=float, default=50.0, help="Geschwindigkeit")
    ap.add_argument("--acc", type=float, default=200.0, help="Beschleunigung")
    args = ap.parse_args()

    # feste Orientierung (top-down) - nimm die Werte, die bei dir funktionieren
    RX = -180.0
    RY = -70.0
    RZ = -65.0

    A, b = load_table_to_robot(args.calib)
    Xr, Yr = map_table_to_robot(A, b, args.xt, args.yt)

    print("=== move_to_table_xy ===")
    print(f"Table target:  Xt={args.xt:.1f} mm  Yt={args.yt:.1f} mm")
    print(f"Robot target:  X={Xr:.2f} mm  Y={Yr:.2f} mm  Z={args.z:.1f} mm")
    print(f"Using calib: {args.calib}")

    arm = XArmAPI(args.ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)

    code = arm.set_position(
        x=Xr, y=Yr, z=args.z,
        roll=RX, pitch=RY, yaw=RZ,
        speed=args.speed, acc=args.acc,
        wait=True
    )

    if code != 0:
        print("ERROR: set_position failed, code =", code)
    else:
        print("Reached table target (mapped to robot base).")

    arm.disconnect()


if __name__ == "__main__":
    main()
