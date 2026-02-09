#!/usr/bin/env python3
# fit_cup_warp.py
import argparse
import json
from pathlib import Path
import numpy as np

def design_matrix(x, y):
    # quadratic 2D basis: 1, x, y, x^2, x*y, y^2
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=1)

def main():
    ap = argparse.ArgumentParser(description="Fit quadratic warp (x_meas,y_meas)->(x_true,y_true)")
    ap.add_argument("--in_json", required=True, help="Path to calibration points json")
    ap.add_argument("--out_npz", default="cup_warp.npz", help="Output npz with cx,cy")
    args = ap.parse_args()

    pts = json.loads(Path(args.in_json).read_text())
    if len(pts) < 6:
        raise SystemExit("Need at least 6 points for quadratic fit; use 9-16 points recommended")

    x = np.array([p["x_meas"] for p in pts], dtype=np.float64)
    y = np.array([p["y_meas"] for p in pts], dtype=np.float64)
    xt = np.array([p["x_true"] for p in pts], dtype=np.float64)
    yt = np.array([p["y_true"] for p in pts], dtype=np.float64)

    A = design_matrix(x, y)

    cx, *_ = np.linalg.lstsq(A, xt, rcond=None)
    cy, *_ = np.linalg.lstsq(A, yt, rcond=None)

    predx = A @ cx
    predy = A @ cy
    err = np.sqrt((predx - xt) ** 2 + (predy - yt) ** 2)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mx = float(np.max(err))

    print(f"points: {len(pts)}")
    print(f"RMSE: {rmse:.2f} mm")
    print(f"max:  {mx:.2f} mm")

    np.savez(args.out_npz, cx=cx.astype(np.float64), cy=cy.astype(np.float64))
    print(f"saved -> {args.out_npz}")

if __name__ == "__main__":
    main()
