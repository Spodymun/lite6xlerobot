import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Marker -> Tisch (mm)
# ----------------------------
MARKER_TABLE_MM = {
    10: (0.0,    0.0),
    11: (720.0,  0.0),
    13: (0.0,  1100.0),
    15: (720.0,1100.0),

    # optional mehr Punkte für Robustheit:
    12: (0.0,   500.0),
    14: (720.0, 500.0),
    16: (360.0, 250.0),
    17: (360.0, 800.0),
}
REQUIRED_CORNER_IDS = [10, 11, 13, 15]
RANSAC_THRESH = 2.0

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


def marker_center(corners_4x2: np.ndarray) -> np.ndarray:
    return corners_4x2.mean(axis=0)


def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt = np.array([[pixel_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]


def compute_homography(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, corners, ids, "No markers"

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


def is_black_roi(frame_bgr, x1, y1, x2, y2, v_mean_max=80, shrink=0.2):
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False, 999.0

    # shrink box
    bw, bh = (x2 - x1), (y2 - y1)
    dx, dy = int(bw * shrink), int(bh * shrink)
    sx1, sy1, sx2, sy2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    sx1 = max(0, min(w - 1, sx1)); sx2 = max(0, min(w, sx2))
    sy1 = max(0, min(h - 1, sy1)); sy2 = max(0, min(h, sy2))
    if sx2 <= sx1 or sy2 <= sy1:
        return False, 999.0

    roi = frame_bgr[sy1:sy2, sx1:sx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[:, :, 2].mean())
    return (v_mean <= v_mean_max), v_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (0/1/2...)")

    # ✅ Default jetzt dein trainiertes Modell:
    ap.add_argument(
        "--weights",
        type=str,
        default="/home/nelly/src/lite6xlerobot/ball_erkennung/runs/detect/train/weights/best.pt",
        help="YOLO weights path"
    )

    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")

    # ✅ Klasse jetzt 'ball' (nicht mehr 'sports ball'):
    ap.add_argument("--ball-class", type=str, default="ball",
                    help="YOLO class name to treat as ball")

    # Schwarz-Check optional machen
    ap.add_argument("--use-black-check", action="store_true",
                    help="Enable black ROI check (HSV V-mean) as extra filter")
    ap.add_argument("--black-vmax", type=float, default=80, help="Max mean V (HSV) to accept as black")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Force inference device. Use cpu if CUDA arch mismatch.")
    args = ap.parse_args()

    print("INFO: q/ESC zum Beenden")
    print(f"INFO: cam={args.cam} weights={args.weights} device={args.device} conf={args.conf} class={args.ball_class}")
    print(f"INFO: black_check={'ON' if args.use_black_check else 'OFF'} (vmax={args.black_vmax})")

    model = YOLO(args.weights)
    if args.device == "cpu":
        model.to("cpu")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera konnte nicht geöffnet werden. --cam {args.cam} prüfen.")

    cv2.namedWindow("view")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()

        # 1) Pixel->Table Homography
        H, aruco_corners, aruco_ids, h_msg = compute_homography(frame)
        if aruco_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, aruco_corners, aruco_ids)

        cv2.putText(vis, h_msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if H is not None else (0, 0, 255), 2)

        # 2) YOLO inference
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
                            # zeichne trotzdem rot, damit du siehst was verworfen wurde
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(vis, f"{cls_name} {conf:.2f} V={v_mean:.0f} (rej)",
                                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            continue

                    # akzeptierte Box zeichnen
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    if v_mean is not None:
                        label += f" V={v_mean:.0f}"
                    cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if best is None or conf > best[0]:
                        best = (conf, x1, y1, x2, y2, v_mean)

        # 3) Best -> center -> table coords
        if best is not None:
            conf, x1, y1, x2, y2, v_mean = best
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(vis, f"BALL px=({cx},{cy}) conf={conf:.2f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if H is not None:
                table_xy = pixel_to_table_xy(np.array([cx, cy], dtype=np.float32), H)
                cv2.putText(vis, f"BALL table_mm=({table_xy[0]:.1f},{table_xy[1]:.1f})",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                msg = f"BALL table_mm=({table_xy[0]:.1f}, {table_xy[1]:.1f}) conf={conf:.2f}"
                if v_mean is not None:
                    msg += f" V={v_mean:.1f}"
                print(msg)
            else:
                cv2.putText(vis, "H not ready", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
