#!/usr/bin/env python3
# cup_warp.py
import numpy as np
from typing import Optional, Tuple

def load_cup_warp_npz(path: str) -> dict:
    """
    Loads a quadratic 2D warp from npz: cx, cy (each shape (6,))
    """
    z = np.load(path)
    cx = z["cx"].astype(np.float64)
    cy = z["cy"].astype(np.float64)
    if cx.shape != (6,) or cy.shape != (6,):
        raise ValueError(f"warp coeffs must be shape (6,), got cx={cx.shape}, cy={cy.shape}")
    return {"cx": cx, "cy": cy}

def apply_cup_warp(x_mm: float, y_mm: float, warp: Optional[dict]) -> Tuple[float, float]:
    """
    Quadratic warp:
      f = [1, x, y, x^2, x*y, y^2]
      x' = f @ cx
      y' = f @ cy
    """
    if warp is None:
        return float(x_mm), float(y_mm)

    x = float(x_mm)
    y = float(y_mm)
    f = np.array([1.0, x, y, x*x, x*y, y*y], dtype=np.float64)
    cx = warp["cx"]
    cy = warp["cy"]
    return float(f @ cx), float(f @ cy)
