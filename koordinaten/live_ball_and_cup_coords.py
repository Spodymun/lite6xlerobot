#!/usr/bin/env python3
"""
live_ball_and_cup_coords.py

- CUP: pixel -> table_mm via H, danach optional cup_warp (mm_measured -> mm_true)
- BALL: pixel -> table_mm via H, KEIN warp

Machine-readable stdout (nur wenn erkannt):
  CUP_MM_WARPED x y conf
  BALL_MM_NOWARP x y conf

Beenden: q oder ESC
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cup_warp import load_cup_warp_npz, apply_cup_warp


def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt = np.array([[pixel_xy]], dtype=np.float32)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]  # (2,)


def best_box_for_class(result, class_name: str):
    """
    Returns best box for a given YOLO class_name:
      (conf, x1,y1,x2,y2, cls_name)
    or None
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    names = result.names
    best = None
    for b in result.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        cls = str(names.get(cls_id, cls_id))
        if cls != class_name:
            continue

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cand = (conf, x1, y1, x2, y2, cls)
        if best is None or conf > best[0]:
            best = cand
    return best


def draw_box(vis, box, color, label_prefix):
    conf, x1, y1, x2, y2, cls = box
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        vis,
        f"{label_prefix} {cls} {conf:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
    return cx, cy, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)

    ap.add_argument("--H", type=str, default="H.npy", help="3x3 homography pixel->table(mm)")
    ap.add_argument(
        "--weights",
        type=str,
        default="/home/nelly/src/lite6xlerobot/ball_erkennung/ball_and_cups.v1i.yolov8/runs/detect/train2/weights/best.pt",
    )
    ap.add_argument("--conf", type=float, default=0.35)

    ap.add_argument("--ball-class", type=str, default="ball")
    ap.add_argument("--cup-class", type=str, default="cup")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Warp nur für CUP
    ap.add_argument("--cup_warp_npz", type=str, default="", help="optional cup warp npz (cx,cy)")

    # Optional: Bild spiegeln / drehen falls nötig
    ap.add_argument("--flip", type=int, default=-1, help="-1: none, 0: vertical, 1: horizontal, 2: both")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])

    args = ap.parse_args()

    print("INFO: q/ESC zum Beenden")
    print(f"INFO: cam={args.cam} weights={args.weights} device={args.device} conf={args.conf}")
    print(f"INFO: H={args.H} (fixed)")
    print(f"INFO: ball_class={args.ball_class} cup_class={args.cup_class}")

    # Load H
    h_path = Path(args.H)
    if not h_path.exists():
        raise RuntimeError(f"H file not found: {h_path.resolve()}")
    H_fixed = np.load(str(h_path))
    if getattr(H_fixed, "shape", None) != (3, 3):
        raise RuntimeError(f"H must be 3x3 .npy, got shape={getattr(H_fixed,'shape',None)}")

    # Load cup warp
    cup_warp = None
    if args.cup_warp_npz:
        wp = Path(args.cup_warp_npz)
        if not wp.exists():
            raise RuntimeError(f"cup_warp_npz not found: {wp.resolve()}")
        cup_warp = load_cup_warp_npz(str(wp))
        if "cx" not in cup_warp or "cy" not in cup_warp:
            raise RuntimeError(f"warp_npz missing cx/cy arrays: keys={list(cup_warp.keys())}")
        print(f"INFO: CUP warp enabled: {wp}")

    # Load YOLO
    model = YOLO(args.weights)
    if args.device == "cpu":
        model.to("cpu")

    # Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera konnte nicht geöffnet werden. --cam {args.cam} prüfen.")

    cv2.namedWindow("view")

    def apply_view_transforms(frame):
        if args.flip in (0, 1):
            frame = cv2.flip(frame, args.flip)
        elif args.flip == 2:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = apply_view_transforms(frame)
        vis = frame.copy()

        cv2.putText(vis, f"H: {h_path.name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"CUP warp: {'ON' if cup_warp is not None else 'OFF'}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0) if cup_warp is not None else (128, 128, 128),
            2,
        )

        # One YOLO pass for both classes
        results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)
        r0 = results[0] if results else None

        cup_box = best_box_for_class(r0, args.cup_class)
        ball_box = best_box_for_class(r0, args.ball_class)

        y_cursor = 90

        # CUP (with warp)
        if cup_box is not None:
            cx, cy, conf = draw_box(vis, cup_box, (0, 255, 0), "CUP")
            cup_mm = pixel_to_table_xy(np.array([cx, cy], dtype=np.float32), H_fixed)
            cup_mm_w = cup_mm
            if cup_warp is not None:
                xw, yw = apply_cup_warp(float(cup_mm[0]), float(cup_mm[1]), cup_warp)
                cup_mm_w = np.array([xw, yw], dtype=np.float32)

            cv2.putText(
                vis,
                f"CUP px=({cx},{cy})",
                (20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y_cursor += 26
            cv2.putText(
                vis,
                f"CUP table_mm (warped)=({cup_mm_w[0]:.1f},{cup_mm_w[1]:.1f})",
                (20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y_cursor += 30

            print(f"CUP_MM_WARPED {cup_mm_w[0]:.1f} {cup_mm_w[1]:.1f} {conf:.3f}")
        else:
            cv2.putText(vis, "No cup detection", (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y_cursor += 30

        # BALL (no warp)
        if ball_box is not None:
            cx, cy, conf = draw_box(vis, ball_box, (255, 255, 0), "BALL")
            ball_mm = pixel_to_table_xy(np.array([cx, cy], dtype=np.float32), H_fixed)

            cv2.putText(
                vis,
                f"BALL px=({cx},{cy})",
                (20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y_cursor += 26
            cv2.putText(
                vis,
                f"BALL table_mm (no-warp)=({ball_mm[0]:.1f},{ball_mm[1]:.1f})",
                (20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y_cursor += 30

            print(f"BALL_MM_NOWARP {ball_mm[0]:.1f} {ball_mm[1]:.1f} {conf:.3f}")
        else:
            cv2.putText(vis, "No ball detection", (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y_cursor += 30

        cv2.imshow("view", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
