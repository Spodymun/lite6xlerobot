#!/usr/bin/env python3
import sys
import time
import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# xArm SDK
from xarm.wrapper import XArmAPI

ROBOT_IP = "10.77.77.200"   # <-- anpassen falls nötig
SET_MODE = True            # True = Skript setzt mode/state, False = nur lesen
MODE = 0                   # 0 = position control (passt bei dir)
STATE = 0                  # 0 = ready

@dataclass
class Pair:
    table_xy_mm: Tuple[float, float]
    robot_xyz_mm: Tuple[float, float, float]

def fit_affine_2d(pairs: List[Pair]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit Robot_XY = A * Table_XY + b using least squares.
    Returns A (2x2), b (2,), rmse_mm
    """
    if len(pairs) < 3:
        raise ValueError("Mindestens 3 Punktpaare nötig (besser 6-12).")

    M = []
    t = []
    for p in pairs:
        x, y = p.table_xy_mm
        X, Y, _Z = p.robot_xyz_mm
        M.append([x, y, 1.0, 0.0, 0.0, 0.0])
        M.append([0.0, 0.0, 0.0, x, y, 1.0])
        t.append(X)
        t.append(Y)

    M = np.array(M, dtype=np.float64)
    t = np.array(t, dtype=np.float64)

    params, *_ = np.linalg.lstsq(M, t, rcond=None)
    a11, a12, b1, a21, a22, b2 = params

    A = np.array([[a11, a12],
                  [a21, a22]], dtype=np.float64)
    b = np.array([b1, b2], dtype=np.float64)

    errs = []
    for p in pairs:
        x, y = p.table_xy_mm
        pred = A @ np.array([x, y]) + b
        X, Y, _Z = p.robot_xyz_mm
        errs.append(np.linalg.norm(pred - np.array([X, Y])))

    rmse = float(np.sqrt(np.mean(np.square(errs))))
    return A, b, rmse

def parse_xy(prompt: str) -> Optional[Tuple[float, float]]:
    s = input(prompt).strip()
    if s == "":
        return None
    s = s.replace(";", ",").replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
        print("Bitte als x,y eingeben (z.B. 150,200).")
        return parse_xy(prompt)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("Konnte Zahlen nicht lesen.")
        return parse_xy(prompt)

def main():
    print("\n=== Tisch -> Roboter Kalibrierung (LIVE TCP via xArm SDK) ===")
    print("Einheit: mm")
    print(f"Robot IP: {ROBOT_IP}\n")

    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    if SET_MODE:
        arm.motion_enable(True)
        arm.set_mode(MODE)
        arm.set_state(STATE)
        time.sleep(0.2)

    print("Hinweise:")
    print("  - Fahre den TCP (Tool Center Point) nacheinander über mehrere Tischpunkte.")
    print("  - Halte Z möglichst konstant (z.B. 30-50mm über Tisch).")
    print("  - Punkte gut verteilen (Ecken + innen).")
    print("  - Eingabeformat Tischpunkt: x,y (mm). Enter = fertig.\n")

    pairs: List[Pair] = []

    while True:
        table_xy = parse_xy(f"[{len(pairs)+1}] Tischpunkt x,y in mm (Enter = fertig): ")
        if table_xy is None:
            break

        input("    Roboter jetzt über diesen Punkt fahren. Dann Enter drücken zum Auslesen...")

        code, pos = arm.get_position()
        if code != 0 or pos is None:
            print(f"    FEHLER: get_position() ret={code}, pos={pos}")
            continue

        X, Y, Z, Rx, Ry, Rz = pos
        print(f"    TCP gelesen: X={X:.3f}  Y={Y:.3f}  Z={Z:.3f} (mm)")
        pairs.append(Pair(table_xy_mm=table_xy, robot_xyz_mm=(float(X), float(Y), float(Z))))
        print("    OK gespeichert.\n")

    if len(pairs) < 3:
        print("Zu wenige Punkte. Mindestens 3, besser 6-12.")
        arm.disconnect()
        sys.exit(1)

    A, b, rmse = fit_affine_2d(pairs)

    print("\n=== Ergebnis (Robot_XY = A * Table_XY + b) ===")
    print(f"A = [[{A[0,0]: .6f}, {A[0,1]: .6f}],")
    print(f"     [{A[1,0]: .6f}, {A[1,1]: .6f}]]")
    print(f"b = [{b[0]: .3f}, {b[1]: .3f}]  mm")
    print(f"RMSE: {rmse:.2f} mm  (Richtwert: <5mm sehr gut, 5-10mm ok)\n")

    # Option: Z prüfen
    zs = [p.robot_xyz_mm[2] for p in pairs]
    print(f"Z-Range: min={min(zs):.2f} mm  max={max(zs):.2f} mm  (kleiner Bereich = besser)\n")

    out = {
        "unit": "mm",
        "model": "affine_2d_table_to_robot_xy",
        "robot_ip": ROBOT_IP,
        "A": A.tolist(),
        "b": b.tolist(),
        "rmse_mm": rmse,
        "z_mm_mean": float(np.mean(zs)),
        "z_mm_min": float(min(zs)),
        "z_mm_max": float(max(zs)),
        "pairs_used": [
            {
                "table_xy_mm": [p.table_xy_mm[0], p.table_xy_mm[1]],
                "robot_xyz_mm": [p.robot_xyz_mm[0], p.robot_xyz_mm[1], p.robot_xyz_mm[2]],
            }
            for p in pairs
        ],
        "notes": {
            "how_to_use": "robot_xy = A @ table_xy + b",
            "tip": "Mehr Punkte + bessere Verteilung = stabiler. Z konstant halten.",
        }
    }

    out_path = "table_to_robot.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"Gespeichert: {out_path}")

    arm.disconnect()

if __name__ == "__main__":
    main()
