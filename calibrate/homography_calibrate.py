#!/usr/bin/env python3
"""
Calibrate fixed homography H.npy using ArUco markers (Pixel -> Table mm)

Setup (your 8 markers):
- id 10 -> (0, 0)
- id 11 -> (720, 0)
- id 12 -> (0, 500)
- id 13 -> (0, 1100)
- id 14 -> (720, 500)
- id 15 -> (720, 1100)
- id 16 -> (360, 250)
- id 17 -> (360, 800)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# =========================
# Marker -> Table (mm)
# =========================
MARKER_TABLE_MM = {
    10: (0.0,    0.0),
    11: (720.0,  0.0),
    12: (0.0,  500.0),
    13: (0.0, 1100.0),
    14: (720.0, 500.0),
    15: (720.0,1100.0),
    16: (360.0, 250.0),
    17: (360.0, 800.0),
}

# Require the 4 corner markers to be visible for a valid H
REQUIRED_CORNER_IDS = [10, 11, 13, 15]

# RANSAC reprojection threshold (in pixels)
RANSAC_THRESH = 2.0


def marker_center(corners_4x2: np.ndarray) -> np.ndarray:
    """corners_4x2: shape (4,2) -> center (2,)"""
    return corners_4x2.mean(axis=0)


def compute_homography_from_markers(frame_bgr: np.ndarray, detector):
    """
    Returns:
      H (3x3) or None,
      corners, ids (as returned by detectMarkers),
      status message
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, corners, ids, "No markers detected"

    ids_flat = ids.flatten().tolist()

    if not all(mid in ids_flat for mid in REQUIRED_CORNER_IDS):
        return None, corners, ids, "Need corner IDs 10,11,13,15 visible"

    pixel_pts, table_pts = [], []
    for mid, (tx, ty) in MARKER_TABLE_MM.items():
        if mid not in ids_flat:
            continue
        idx = ids_flat.index(mid)
        c = corners[idx].reshape(4, 2).astype(np.float32)
        center = marker_center(c)
        pixel_pts.append(center)
        table_pts.append([tx, ty])
        print("Marker", mid)
        print(c)

    if len(pixel_pts) < 4:
        return None, corners, ids, "Need >=4 known markers"

    H, inliers = cv2.findHomography(
        np.array(pixel_pts, dtype=np.float32),
        np.array(table_pts, dtype=np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH
    )
    if H is None:
        return None, corners, ids, "Homography failed"

    inl = int(inliers.sum()) if inliers is not None else len(pixel_pts)
    return H, corners, ids, f"H OK points={len(pixel_pts)} inliers={inl}"


def main():
    ap = argparse.ArgumentParser(description="Calibrate H.npy from ArUco markers (pixel->table mm)")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (0/1/2...)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--dict", type=str, default="DICT_4X4_50",
                    help="Aruco dict name, e.g. DICT_4X4_50")
    ap.add_argument("--out", type=str, default="H.npy", help="Output file (H.npy)")
    ap.add_argument("--stable-frames", type=int, default=10,
                    help="Auto-save after H is stable for N frames")
    ap.add_argument("--stable-eps", type=float, default=1e-3,
                    help="Mean abs diff threshold between consecutive H for stability")

    args = ap.parse_args()

    # ArUco setup
    if not hasattr(cv2.aruco, args.dict):
        raise RuntimeError(f"Unknown aruco dict: {args.dict}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Camera
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: --cam {args.cam}")

    out_path = Path(args.out)

    last_H = None
    stable_count = 0
    best_H = None  # last valid H

    print("Controls:")
    print("  s = save H now")
    print("  q / ESC = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()

        H, aruco_corners, aruco_ids, msg = compute_homography_from_markers(vis, detector)

        # draw markers if any
        if aruco_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, aruco_corners, aruco_ids)

        # stability check
        if H is not None:
            best_H = H
            if last_H is None:
                stable_count = 1
            else:
                diff = float(np.mean(np.abs(H - last_H)))
                if diff <= args.stable_eps:
                    stable_count += 1
                else:
                    stable_count = 1
            last_H = H

            cv2.putText(
                vis,
                f"{msg}  stable={stable_count}/{args.stable_frames}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # auto-save if stable long enough
            if stable_count >= args.stable_frames:
                np.save(str(out_path), best_H)
                cv2.putText(
                    vis,
                    f"SAVED -> {out_path}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("calibrate_h_aruco", vis)
                cv2.waitKey(500)
                break
        else:
            stable_count = 0
            last_H = None
            cv2.putText(
                vis,
                msg,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("calibrate_h_aruco", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and best_H is not None:
            np.save(str(out_path), best_H)
            print(f"Saved H to {out_path.resolve()}")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()