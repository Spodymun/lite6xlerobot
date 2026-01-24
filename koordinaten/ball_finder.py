#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    pixel_xy: np.array([x_px, y_px], dtype=float32)
    H: 3x3 homography pixel -> table(mm)
    returns: np.array([x_mm, y_mm], dtype=float32)
    """
    pt = np.array([[pixel_xy]], dtype=np.float32)  # shape (1,1,2)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]


def is_black_roi(frame_bgr, x1, y1, x2, y2, v_mean_max=80, shrink=0.2):
    """
    Extra Filter: akzeptiere Box nur, wenn ROI im HSV-V-Kanal dunkel genug ist.
    Hilft gegen False-Positives.
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False, 999.0

    # shrink box
    bw, bh = (x2 - x1), (y2 - y1)
    dx, dy = int(bw * shrink), int(bh * shrink)
    sx1, sy1, sx2, sy2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    sx1 = max(0, min(w - 1, sx1))
    sx2 = max(0, min(w, sx2))
    sy1 = max(0, min(h - 1, sy1))
    sy2 = max(0, min(h, sy2))
    if sx2 <= sx1 or sy2 <= sy1:
        return False, 999.0

    roi = frame_bgr[sy1:sy2, sx1:sx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[:, :, 2].mean())
    return (v_mean <= v_mean_max), v_mean


def main():
    ap = argparse.ArgumentParser(description="Ball detection -> table coordinates (mm) using fixed H.npy")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (0/1/2...)")

    ap.add_argument("--H", type=str, default="H.npy",
                    help="Path to fixed homography (pixel->table mm)")
    ap.add_argument("--once", action="store_true",
                    help="Exit after first valid BALL_MM output")

    ap.add_argument("--weights", type=str,
                    default="/home/nelly/src/lite6xlerobot/ball_erkennung/runs/detect/train/weights/best.pt",
                    help="YOLO weights path")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")

    ap.add_argument("--ball-class", type=str, default="ball",
                    help="YOLO class name to treat as ball")

    ap.add_argument("--use-black-check", action="store_true",
                    help="Enable black ROI check (HSV V-mean) as extra filter")
    ap.add_argument("--black-vmax", type=float, default=80,
                    help="Max mean V (HSV) to accept as black")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Force inference device. Use cpu if CUDA arch mismatch.")

    args = ap.parse_args()

    print("INFO: q/ESC zum Beenden")
    print(f"INFO: cam={args.cam} weights={args.weights} device={args.device} conf={args.conf} class={args.ball_class}")
    print(f"INFO: H={args.H} (fixed)")
    print(f"INFO: black_check={'ON' if args.use_black_check else 'OFF'} (vmax={args.black_vmax})")

    # Load H
    h_path = Path(args.H)
    if not h_path.exists():
        raise RuntimeError(f"H file not found: {h_path.resolve()}")
    H_fixed = np.load(str(h_path))
    if H_fixed.shape != (3, 3):
        raise RuntimeError(f"H must be 3x3, got shape={H_fixed.shape}")

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
        cv2.putText(vis, "H: fixed (H.npy)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # YOLO inference
        results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)

        best = None  # (conf, x1,y1,x2,y2, v_mean_or_none)
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

                    v_mean = None
                    if args.use_black_check:
                        black_ok, v_mean = is_black_roi(
                            frame, x1, y1, x2, y2,
                            v_mean_max=args.black_vmax, shrink=0.2
                        )
                        if not black_ok:
                            # show rejected
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                vis,
                                f"{cls_name} {conf:.2f} V={v_mean:.0f} (rej)",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )
                            continue

                    # accepted box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    if v_mean is not None:
                        label += f" V={v_mean:.0f}"
                    cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if best is None or conf > best[0]:
                        best = (conf, x1, y1, x2, y2, v_mean)

        # Best -> center -> table coords
        if best is not None:
            conf, x1, y1, x2, y2, v_mean = best
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(vis, f"BALL px=({cx},{cy}) conf={conf:.2f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            table_xy = pixel_to_table_xy(np.array([cx, cy], dtype=np.float32), H)
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