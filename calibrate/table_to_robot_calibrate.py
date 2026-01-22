#!/usr/bin/env python3
"""
table_to_robot_calibrate.py

Calibrates an affine mapping from table coordinates (mm) to robot base XY (mm):
    [Xr, Yr]^T = A * [Xt, Yt]^T + b

Workflow:
1) Define a list of table points in mm (Xt, Yt) you can physically reach/align with TCP.
2) For each point:
   - Move robot TCP above that point (by eye, from top view) at constant Z
   - Press ENTER
   - Script reads current TCP pose via xArm get_position()
3) Fit affine transform, report RMS error, save to YAML.

Run:
  python3 table_to_robot_calibrate.py --ip 10.77.77.200 --out table_to_robot.yaml

Note:
- This does NOT need the arm marker in view.
- Table points can be marker centers (you already have their mm positions).
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import yaml

# --- xArm SDK ---
from xarm.wrapper import XArmAPI


# ----------------------------
# Put your table points here
# (use points you can align TCP above)
# ----------------------------
TABLE_POINTS_MM: List[Tuple[float, float]] = [
    # Ecken + innen (Beispiele: nutze deine Marker oder gut erreichbare Punkte)
    (720.0, 800.0),        # 1
    (590.0, 1000.0),      # 2
    (515.0, 850.0),     # 3
    (570.0, 750.0),   # 4
    (465.0, 1050.0),    # 5
    (360.0, 1100.0),    # 6
    (360.0, 900.0),      # 7
    (265.0, 1000.0),    # 8
    (165.0, 950.0),      # 9
    (240.0, 800.0),    # 10
]


@dataclass
class Sample:
    table_xy: Tuple[float, float]
    robot_xy: Tuple[float, float]
    robot_pose6: Tuple[float, float, float, float, float, float]


def fit_affine_2d(table_xy: np.ndarray, robot_xy: np.ndarray):
    """
    Solve for affine mapping:
      Xr = a11*Xt + a12*Yt + b1
      Yr = a21*Xt + a22*Yt + b2
    """
    assert table_xy.shape[1] == 2
    assert robot_xy.shape[1] == 2
    n = table_xy.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for affine fit (better 6-10).")

    # Build design matrix: [Xt Yt 1]
    M = np.hstack([table_xy, np.ones((n, 1), dtype=np.float64)])  # (n,3)

    # Solve least squares for X and Y separately
    # params_x = [a11 a12 b1]
    # params_y = [a21 a22 b2]
    params_x, *_ = np.linalg.lstsq(M, robot_xy[:, 0], rcond=None)
    params_y, *_ = np.linalg.lstsq(M, robot_xy[:, 1], rcond=None)

    A = np.array([[params_x[0], params_x[1]],
                  [params_y[0], params_y[1]]], dtype=np.float64)
    b = np.array([params_x[2], params_y[2]], dtype=np.float64)

    # Predict + errors
    pred = (M @ np.vstack([params_x, params_y]).T)  # (n,2)
    err = pred - robot_xy
    rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    per_point = np.sqrt(np.sum(err**2, axis=1)).astype(float)

    return A, b, rmse, per_point, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, required=True, help="Robot IP (xArm), e.g. 10.77.77.200")
    ap.add_argument("--out", type=str, default="table_to_robot.yaml")
    ap.add_argument("--speed", type=float, default=50.0, help="(optional) for future auto moves, not used here")
    ap.add_argument("--acc", type=float, default=200.0, help="(optional) for future auto moves, not used here")
    args = ap.parse_args()

    arm = XArmAPI(args.ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)  # position control with adjustable speed/acc
    arm.set_state(0)
    time.sleep(0.5)

    print("\n=== table_to_robot_calibrate ===")
    print("TCP pose is read via xArm get_position() in mm/deg.")
    print("For each table point, align TCP ABOVE the point (top-down) at constant Z, then press ENTER.\n")

    samples: List[Sample] = []

    for i, (xt, yt) in enumerate(TABLE_POINTS_MM):
        input(f"[{i+1}/{len(TABLE_POINTS_MM)}] Move TCP above TABLE point (Xt,Yt)=({xt:.1f},{yt:.1f}) mm and press ENTER...")

        code, pose = arm.get_position(is_radian=False)
        if code != 0 or pose is None:
            print("ERROR: get_position failed, code=", code, "pose=", pose)
            continue

        # pose: [x, y, z, roll, pitch, yaw] in mm/deg
        xr, yr, zr, rx, ry, rz = pose
        print(f"  Read TCP: X={xr:.3f} Y={yr:.3f} Z={zr:.3f} Rx={rx:.3f} Ry={ry:.3f} Rz={rz:.3f}")

        samples.append(Sample(
            table_xy=(float(xt), float(yt)),
            robot_xy=(float(xr), float(yr)),
            robot_pose6=(float(xr), float(yr), float(zr), float(rx), float(ry), float(rz)),
        ))

    if len(samples) < 3:
        raise SystemExit("Not enough samples collected (<3).")

    table_xy = np.array([s.table_xy for s in samples], dtype=np.float64)
    robot_xy = np.array([s.robot_xy for s in samples], dtype=np.float64)

    A, b, rmse, per_point, pred = fit_affine_2d(table_xy, robot_xy)

    print("\n=== FIT RESULT ===")
    print("A =\n", A)
    print("b =", b)
    print(f"RMSE (mm): {rmse:.3f}")
    print("Per-point error (mm):", [float(x) for x in per_point])

    # Save YAML
    out = {
        "model": "affine_2d",
        "units": {"table": "mm", "robot": "mm"},
        "A": A.tolist(),
        "b": b.tolist(),
        "rmse_mm": rmse,
        "samples": [
            {
                "table_mm": [s.table_xy[0], s.table_xy[1]],
                "robot_mm": [s.robot_xy[0], s.robot_xy[1]],
                "robot_pose6": list(s.robot_pose6),
            }
            for s in samples
        ],
    }

    with open(args.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"\nSaved: {args.out}")

    arm.disconnect()


if __name__ == "__main__":
    main()
