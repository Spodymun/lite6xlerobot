#!/usr/bin/env python3
"""
Beer-Pong Cup Detection (YOLO + Aimpoint checks) - CLEAN + NEAREST-TO-ARM BIAS

Neu:
- Becher-Auswahl bevorzugt Ziele, die näher am Arm liegen (ARM_MM), ohne den visuellen Score zu ignorieren.
- Auswahlkriterium: score - DIST_W * distance_to_arm_mm

Machine-readable output (only when READY):
  CUP_MM x_mm y_mm score yolo_conf

Beispiel:
  python3 cups_yolo.py --cam 2 --H H.npy --device cpu --once
"""

import argparse
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ==============================
# Data structure
# ==============================
@dataclass
class Det:
    rim_px: Tuple[float, float]
    r_px: float
    x_mm: float
    y_mm: float
    diam_mm: float
    red_ratio: float
    black_frac: float
    white_inner: float
    score: float
    yolo_conf: float
    box_xyxy: Tuple[int, int, int, int]


# ==============================
# Geometry / transforms
# ==============================
def pix_to_table(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def estimate_diameter_mm_from_homography(H: np.ndarray, cx: float, cy: float, r_px: float) -> float:
    x0, y0 = pix_to_table(H, cx, cy)
    xr, yr = pix_to_table(H, cx + r_px, cy)
    xd, yd = pix_to_table(H, cx, cy + r_px)
    r1 = np.hypot(xr - x0, yr - y0)
    r2 = np.hypot(xd - x0, yd - y0)
    r_mm = 0.5 * (r1 + r2)
    return float(2.0 * r_mm)


def in_mm_roi(x_mm: float, y_mm: float, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    return (xmin <= x_mm <= xmax) and (ymin <= y_mm <= ymax)


def clip_roi(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(1, min(w, int(x1)))
    y1 = max(1, min(h, int(y1)))
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return x0, y0, x1, y1


# ==============================
# Masks / checks
# ==============================
def red_mask_hsv(hsv: np.ndarray) -> np.ndarray:
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )


def make_disk_mask(shape_hw: Tuple[int, int], center: Tuple[float, float], r: int) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(m, (cx, cy), max(1, r), 255, -1)
    return m


def make_annulus_mask(shape_hw: Tuple[int, int], center: Tuple[float, float], r_in: int, r_out: int) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(m, (cx, cy), max(1, r_out), 255, -1)
    cv2.circle(m, (cx, cy), max(1, r_in), 0, -1)
    return m


def red_ratio_in_annulus(hsv: np.ndarray, center: Tuple[float, float], r_in: int, r_out: int) -> float:
    rm = red_mask_hsv(hsv)
    ann = make_annulus_mask(rm.shape[:2], center, r_in, r_out)
    denom = cv2.countNonZero(ann)
    if denom <= 0:
        return 0.0
    red_px = cv2.countNonZero(cv2.bitwise_and(rm, rm, mask=ann))
    return float(red_px) / float(denom)


def red_ratio_in_disk(hsv: np.ndarray, center: Tuple[float, float], r: int) -> float:
    rm = red_mask_hsv(hsv)
    disk = make_disk_mask(rm.shape[:2], center, r)
    denom = cv2.countNonZero(disk)
    if denom <= 0:
        return 0.0
    red_px = cv2.countNonZero(cv2.bitwise_and(rm, rm, mask=disk))
    return float(red_px) / float(denom)


def white_ratio_in_disk(hsv: np.ndarray, center: Tuple[float, float], r: int, *, v_min: int, s_max: int) -> float:
    disk = make_disk_mask(hsv.shape[:2], center, r)
    denom = cv2.countNonZero(disk)
    if denom <= 0:
        return 0.0
    lower = np.array([0, 0, v_min], dtype=np.uint8)
    upper = np.array([180, s_max, 255], dtype=np.uint8)
    wmask = cv2.inRange(hsv, lower, upper)
    w_px = cv2.countNonZero(cv2.bitwise_and(wmask, wmask, mask=disk))
    return float(w_px) / float(denom)


def black_ring_frac(gray: np.ndarray, center: Tuple[float, float], r: int, *, thick: int, dark_thresh: int) -> float:
    h, w = gray.shape[:2]
    cx, cy = center
    total = 0
    dark = 0
    n_samples = 72
    for k in range(n_samples):
        ang = (2.0 * np.pi * k) / n_samples
        ux = np.cos(ang)
        uy = np.sin(ang)
        for rr in range(max(1, r - thick), r + thick + 1):
            x = int(round(cx + ux * rr))
            y = int(round(cy + uy * rr))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if gray[y, x] <= dark_thresh:
                    dark += 1
    if total == 0:
        return 0.0
    return float(dark) / float(total)


# ==============================
# ROI aimpoint
# ==============================
def find_white_centroid_roi(hsv_roi: np.ndarray, *, v_min: int, s_max: int) -> Optional[Tuple[float, float]]:
    lower = np.array([0, 0, v_min], dtype=np.uint8)
    upper = np.array([180, s_max, 255], dtype=np.uint8)
    wmask = cv2.inRange(hsv_roi, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wmask = cv2.morphologyEx(wmask, cv2.MORPH_OPEN, k, iterations=1)
    wmask = cv2.morphologyEx(wmask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 60:
        return None
    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)


def pick_best_circle_in_roi(
    gray_roi: np.ndarray,
    hsv_roi: np.ndarray,
    *,
    minDist: float,
    param1: int,
    param2: int,
    min_r: int,
    max_r: int,
    ann_in: int,
    ann_out: int,
    inner_r_frac: float,
    white_inner_min: float,
    red_inner_max: float,
    white_v_min: int,
    white_s_max: int,
    red_ring_min: float,
    black_min: float,
    black_thick: int,
    black_dark_thresh: int,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    g = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    best = None
    best_score = -1e9

    for (cx, cy, r) in circles:
        if r <= 0:
            continue

        bfrac = black_ring_frac(gray_roi, (float(cx), float(cy)), int(r),
                                thick=black_thick, dark_thresh=black_dark_thresh)
        if bfrac < black_min:
            continue

        rratio = red_ratio_in_annulus(hsv_roi, (float(cx), float(cy)),
                                      r_in=max(1, int(r + ann_in)),
                                      r_out=max(2, int(r + ann_out)))
        if rratio < red_ring_min:
            continue

        inner_r = max(2, int(r * inner_r_frac))
        w_inner = white_ratio_in_disk(hsv_roi, (float(cx), float(cy)), inner_r,
                                      v_min=white_v_min, s_max=white_s_max)
        if w_inner < white_inner_min:
            continue

        r_inner = red_ratio_in_disk(hsv_roi, (float(cx), float(cy)), inner_r)
        if r_inner > red_inner_max:
            continue

        score = (2.2 * w_inner) + (2.0 * bfrac) + (1.8 * rratio) + (0.01 * r)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r), float(rratio), float(bfrac), float(w_inner))

    return best


# ==============================
# YOLO helper
# ==============================
def yolo_boxes(model, frame_bgr: np.ndarray, *, conf: float, iou: float, cls: Optional[int], device: str):
    res = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False, device=device)[0]
    out = []
    if res.boxes is None:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)

    for (x0, y0, x1, y1), c, k in zip(xyxy, confs, clss):
        if cls is not None and k != cls:
            continue
        out.append((int(x0), int(y0), int(x1), int(y1), float(c), int(k)))
    return out


# ==============================
# Stability helper
# ==============================
def mm_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--H", type=str, default="H.npy", help="Fixed homography pixel->table(mm)")
    ap.add_argument("--once", action="store_true", help="Exit after first READY CUP_MM output")

    # Arm-bias: du kannst das grob setzen, muss nicht mm-genau sein
    ap.add_argument("--arm_x", type=float, default=360.0, help="Arm reference X in table mm")
    ap.add_argument("--arm_y", type=float, default=1300.0, help="Arm reference Y in table mm")
    ap.add_argument("--dist_w", type=float, default=0.002, help="Distance penalty weight (score per mm)")

    # YOLO
    ap.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.45)
    ap.add_argument("--yolo_cls", type=int, default=None)
    ap.add_argument("--roi_pad", type=float, default=0.12)
    ap.add_argument("--device", type=str, default="cpu")

    # mm ROI
    ap.add_argument("--y_offset", type=float, default=50.0)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=800.0)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=1800.0)
    ap.add_argument("--max_diam_mm", type=float, default=100.0)

    # ROI-local Hough
    ap.add_argument("--h_minDist", type=float, default=18.0)
    ap.add_argument("--h_param1", type=int, default=120)
    ap.add_argument("--h_param2", type=int, default=12)
    ap.add_argument("--h_min_r", type=int, default=10)
    ap.add_argument("--h_max_r", type=int, default=70)

    # verification
    ap.add_argument("--red_ring_min", type=float, default=0.12)
    ap.add_argument("--ann_in", type=int, default=2)
    ap.add_argument("--ann_out", type=int, default=14)

    ap.add_argument("--black_min", type=float, default=0.20)
    ap.add_argument("--black_dark_thresh", type=int, default=100)
    ap.add_argument("--black_thick", type=int, default=2)

    ap.add_argument("--inner_r_frac", type=float, default=0.50)
    ap.add_argument("--white_inner_min", type=float, default=0.40)
    ap.add_argument("--red_inner_max", type=float, default=0.15)
    ap.add_argument("--white_v_min", type=int, default=150)
    ap.add_argument("--white_s_max", type=int, default=95)

    # READY logic
    ap.add_argument("--ready_score", type=float, default=2.3)
    ap.add_argument("--ready_frames", type=int, default=2)
    ap.add_argument("--stable_mm", type=float, default=120.0)

    # debug
    ap.add_argument("--show_edges", action="store_true")
    ap.add_argument("--show_red", action="store_true")

    args = ap.parse_args()

    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(args.yolo_model)

    H = np.load(args.H)
    if H.shape != (3, 3):
        raise RuntimeError(f"H must be 3x3, got {H.shape}")

    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: --cam {args.cam}")

    ARM_MM = (float(args.arm_x), float(args.arm_y))
    DIST_W = float(args.dist_w)

    last_best_mm: Optional[Tuple[float, float]] = None
    best_streak = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
        redmask = red_mask_hsv(hsv)

        boxes = yolo_boxes(model, frame, conf=args.yolo_conf, iou=args.yolo_iou, cls=args.yolo_cls, device=args.device)
        dets: List[Det] = []

        for (x0, y0, x1, y1, bconf, cls_id) in boxes:
            bw = x1 - x0
            bh = y1 - y0
            pad_x = int(bw * args.roi_pad)
            pad_y = int(bh * args.roi_pad)

            rx0, ry0, rx1, ry1 = clip_roi(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)
            roi_hsv = hsv[ry0:ry1, rx0:rx1]
            roi_gray = gray[ry0:ry1, rx0:rx1]

            est_r = int(0.25 * min(rx1 - rx0, ry1 - ry0))
            min_r = max(args.h_min_r, int(est_r * 0.6))
            max_r = min(args.h_max_r, int(est_r * 1.5))
            if max_r <= min_r + 2:
                max_r = min(args.h_max_r, min_r + 6)

            best_circle = pick_best_circle_in_roi(
                roi_gray, roi_hsv,
                minDist=args.h_minDist,
                param1=args.h_param1,
                param2=args.h_param2,
                min_r=min_r,
                max_r=max_r,
                ann_in=args.ann_in,
                ann_out=args.ann_out,
                inner_r_frac=args.inner_r_frac,
                white_inner_min=args.white_inner_min,
                red_inner_max=args.red_inner_max,
                white_v_min=args.white_v_min,
                white_s_max=args.white_s_max,
                red_ring_min=args.red_ring_min,
                black_min=args.black_min,
                black_thick=args.black_thick,
                black_dark_thresh=args.black_dark_thresh,
            )

            if best_circle is not None:
                cx, cy, r, rratio, bfrac, w_inner = best_circle
                cx_full = rx0 + cx
                cy_full = ry0 + cy
                r_full = r
            else:
                wc = find_white_centroid_roi(roi_hsv, v_min=args.white_v_min, s_max=args.white_s_max)
                if wc is None:
                    continue

                cx_full = rx0 + wc[0]
                cy_full = ry0 + wc[1]
                r_full = float(max(12, est_r))

                inner_r = max(2, int(r_full * args.inner_r_frac))
                w_inner = white_ratio_in_disk(hsv, (cx_full, cy_full), inner_r,
                                              v_min=args.white_v_min, s_max=args.white_s_max)
                if w_inner < args.white_inner_min:
                    continue

                r_inner = red_ratio_in_disk(hsv, (cx_full, cy_full), inner_r)
                if r_inner > args.red_inner_max:
                    continue

                bfrac = black_ring_frac(gray, (cx_full, cy_full), int(r_full),
                                        thick=args.black_thick, dark_thresh=args.black_dark_thresh)
                if bfrac < args.black_min:
                    continue

                rratio = red_ratio_in_annulus(hsv, (cx_full, cy_full),
                                              r_in=max(1, int(r_full + args.ann_in)),
                                              r_out=max(2, int(r_full + args.ann_out)))
                if rratio < args.red_ring_min:
                    continue

            x_mm, y_mm = pix_to_table(H, float(cx_full), float(cy_full))
            y_mm += args.y_offset
            if not in_mm_roi(x_mm, y_mm, args.xmin, args.xmax, args.ymin, args.ymax):
                continue

            diam_mm = estimate_diameter_mm_from_homography(H, float(cx_full), float(cy_full), float(r_full))
            if diam_mm > args.max_diam_mm:
                continue

            score = (
                (2.2 * float(w_inner)) +
                (2.0 * float(bfrac)) +
                (1.8 * float(rratio)) +
                (0.01 * float(r_full)) -
                (0.002 * float(diam_mm))
            )

            dets.append(Det(
                rim_px=(float(cx_full), float(cy_full)),
                r_px=float(r_full),
                x_mm=float(x_mm),
                y_mm=float(y_mm),
                diam_mm=float(diam_mm),
                red_ratio=float(rratio),
                black_frac=float(bfrac),
                white_inner=float(w_inner),
                score=float(score),
                yolo_conf=float(bconf),
                box_xyxy=(x0, y0, x1, y1),
            ))

        # ---------------------------------------------------------
        # Pick best candidate with NEAREST-TO-ARM bias:
        # best = max(score - DIST_W * distance_to_arm)
        # ---------------------------------------------------------
        def selection_value(d: Det) -> float:
            dist = float(np.hypot(d.x_mm - ARM_MM[0], d.y_mm - ARM_MM[1]))
            return float(d.score) - DIST_W * dist

        best_det: Optional[Det] = max(dets, key=selection_value, default=None)

        # stability streak -> READY
        ready = False
        if best_det is None or best_det.score < args.ready_score:
            best_streak = 0
            last_best_mm = None
        else:
            cur_mm = (best_det.x_mm, best_det.y_mm)
            if last_best_mm is None or mm_dist(cur_mm, last_best_mm) <= args.stable_mm:
                best_streak += 1
            else:
                best_streak = 1
            last_best_mm = cur_mm
            ready = best_streak >= args.ready_frames

        # ------------------ MACHINE OUTPUT ------------------
        if best_det is not None and ready:
            print(f"CUP_MM {best_det.x_mm:.1f} {best_det.y_mm:.1f} {best_det.score:.3f} {best_det.yolo_conf:.3f}")
            if args.once:
                break

        # ------------------ VIS ------------------
        vis = frame.copy()

        # draw YOLO boxes
        for (x0, y0, x1, y1, bconf, cls_id) in boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 180, 0), 2)
            cv2.putText(vis, f"yolo {bconf:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        # draw dets
        for d in dets:
            cx, cy = d.rim_px
            r = int(round(d.r_px))
            cv2.circle(vis, (int(cx), int(cy)), r, (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f"S{d.score:.2f}  W{d.white_inner:.2f} R{d.red_ratio:.2f} B{d.black_frac:.2f}",
                (int(cx) - 85, int(cy) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2
            )

        cv2.putText(
            vis,
            f"ARM=({ARM_MM[0]:.0f},{ARM_MM[1]:.0f})  dist_w={DIST_W:.4f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
        cv2.putText(
            vis,
            f"ROI x[{args.xmin:.0f}..{args.xmax:.0f}] y[{args.ymin:.0f}..{args.ymax:.0f}]  maxDiam={args.max_diam_mm:.0f}mm",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        if best_det is not None:
            cx, cy = best_det.rim_px
            cv2.circle(vis, (int(cx), int(cy)), 14, (0, 0, 255), 3)
            if ready:
                cv2.putText(
                    vis,
                    f"READY streak={best_streak}  THROW ({best_det.x_mm:.1f},{best_det.y_mm:.1f})",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            else:
                cv2.putText(
                    vis,
                    f"NOT READY streak={best_streak}/{args.ready_frames}  bestScore={best_det.score:.2f}/{args.ready_score:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        else:
            cv2.putText(
                vis,
                "NO DETECTIONS",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        cv2.imshow("cups", vis)
        if args.show_edges:
            cv2.imshow("edges", edges)
        if args.show_red:
            cv2.imshow("redmask", redmask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()