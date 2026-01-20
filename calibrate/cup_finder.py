import argparse
import numpy as np
import cv2
import time

def red_mask_hsv(hsv):
    # Rot wrappt um 0°, daher 2 Bereiche
    # Startwerte: funktionieren oft ok; später fein-tunen.
    lower1 = np.array([0,   120, 70], dtype=np.uint8)
    upper1 = np.array([10,  255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)

def clean_mask(mask):
    # Morphology: erst "öffnen" (rauschen weg), dann "schließen" (löcher zu)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] <= 1e-6:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return (cx, cy)

def pix_to_world(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)  # shape (1,1,2)
    out = cv2.perspectiveTransform(pt, H)[0,0]     # (x_mm, y_mm)
    return float(out[0]), float(out[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--min_area", type=int, default=800, help="min contour area in pixels")
    ap.add_argument("--max_cups", type=int, default=4)
    args = ap.parse_args()

    H = np.load(args.H)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = red_mask_hsv(hsv)
        mask = clean_mask(mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cups = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < args.min_area:
                continue
            center = contour_centroid(c)
            if center is None:
                continue
            cx, cy = center
            x_mm, y_mm = pix_to_world(H, cx, cy)
            cups.append({
                "area": float(area),
                "cx": float(cx), "cy": float(cy),
                "x_mm": float(x_mm), "y_mm": float(y_mm),
                "cnt": c,
            })

        # Sort: größte zuerst
        cups.sort(key=lambda d: d["area"], reverse=True)
        cups = cups[:args.max_cups]

        # Visualisieren
        vis = frame.copy()
        for i, d in enumerate(cups):
            c = d["cnt"]
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
            cx, cy = int(d["cx"]), int(d["cy"])
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            label = f"{i}: px=({d['cx']:.0f},{d['cy']:.0f}) mm=({d['x_mm']:.1f},{d['y_mm']:.1f})"
            cv2.putText(vis, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        cv2.imshow("cups", vis)
        cv2.imshow("mask", mask)

        # Console print (5 Hz)
        now = time.time()
        if now - last_print > 0.2:
            last_print = now
            if cups:
                print("CUPS:", ", ".join([f"#{i} ({d['x_mm']:.1f},{d['y_mm']:.1f})mm" for i, d in enumerate(cups)]))
            else:
                print("CUPS: none")

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
