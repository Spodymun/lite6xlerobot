#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Optional warp (measured mm -> true mm)
# Requires cup_warp.py in same folder (or PYTHONPATH)
from cup_warp import load_cup_warp_npz, apply_cup_warp


def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    pixel_xy: np.array([x_px, y_px], dtype=float32)
    H: 3x3 homography pixel -> table(mm)
    returns: np.array([x_mm, y_mm], dtype=float32)
    """
    pt = np.array([[pixel_xy]], dtype=np.float32)  # shape (1,1,2)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]


def main():
    ap = argparse.ArgumentParser(description="Ball detection -> table coordinates (mm) using fixed H.npy (+ optional warp)")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (0/1/2...)")

    ap.add_argument("--H", type=str, default="H.npy",
                    help="Path to fixed homography (pixel->table mm)")
    ap.add_argument("--once", action="store_true",
                    help="Exit after first valid BALL_MM output")

    ap.add_argument("--weights", type=str,
                    default="/home/nelly/src/lite6xlerobot/ball_erkennung/ball_and_cups.v1i.yolov8/runs/detect/train2/weights/best.pt",
                    help="YOLO weights path")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")

    ap.add_argument("--ball-class", type=str, default="ball",
                    help="YOLO class name to treat as ball")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Force inference device. Use cpu if CUDA arch mismatch.")

    # Optional warp: mm_measured -> mm_true (same mapping as for cups if you want it applied)
    ap.add_argument("--warp_npz", type=str, default="",
                    help="Optional mm-warp npz (cx,cy coeffs). If set, warp is applied after H.")

    args = ap.parse_args()

    print("INFO: q/ESC zum Beenden")
    print(f"INFO: cam={args.cam} weights={args.weights} device={args.device} conf={args.conf} class={args.ball_class}")
    print(f"INFO: H={args.H} (fixed)")

    # Load H
    h_path = Path(args.H)
    if not h_path.exists():
        raise RuntimeError(f"H file not found: {h_path.resolve()}")
    H_fixed = np.load(str(h_path))
    if getattr(H_fixed, "shape", None) != (3, 3):
        raise RuntimeError(f"H must be 3x3 .npy, got type={type(H_fixed)} shape={getattr(H_fixed,'shape',None)}")

    # Load optional warp (measured mm -> true mm)
    warp = None
    if args.warp_npz:
        wp = Path(args.warp_npz)
        if not wp.exists():
            raise RuntimeError(f"warp_npz not found: {wp.resolve()}")
        warp = load_cup_warp_npz(str(wp))
        if "cx" not in warp or "cy" not in warp:
            raise RuntimeError(f"warp_npz missing cx/cy arrays: keys={list(warp.keys())}")
        print(f"INFO: warp enabled: {wp} (cx/cy)")

    # Load YOLO
    model = YOLO(args.weights)
    if args.device == "cpu":
        model.to("cpu")

    # Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera konnte nicht geöffnet werden. --cam {args.cam} prüfen.")

    cv2.namedWindow("view")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()

        # Always fixed H
        H = H_fixed
        cv2.putText(vis, f"H: {h_path.name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # YOLO inference
        results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)

        best = None  # (conf, x1,y1,x2,y2)
        if results:
            r = results[0]
            names = r.names
            boxes = r.boxes

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = str(names.get(cls_id, cls_id))

                    if cls_name != args.ball_class:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()

                    # accepted box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if best is None or conf > best[0]:
                        best = (conf, x1, y1, x2, y2)

        # Best -> center -> table coords (+ optional warp)
        if best is not None:
            conf, x1, y1, x2, y2 = best
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(vis, f"BALL px=({cx},{cy}) conf={conf:.2f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            table_xy = pixel_to_table_xy(np.array([cx, cy], dtype=np.float32), H)

            if warp is not None:
                xw, yw = apply_cup_warp(float(table_xy[0]), float(table_xy[1]), warp)
                table_xy = np.array([xw, yw], dtype=np.float32)
                cv2.putText(vis, "warp: ON", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(vis, "warp: OFF", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            cv2.putText(vis, f"BALL table_mm=({table_xy[0]:.1f},{table_xy[1]:.1f})",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Machine-readable output:
            print(f"BALL_MM {table_xy[0]:.1f} {table_xy[1]:.1f} {conf:.3f}")

            if args.once:
                break
        else:
            cv2.putText(vis, "No ball detection", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("view", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
