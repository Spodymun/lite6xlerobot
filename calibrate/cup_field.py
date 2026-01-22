import cv2
import numpy as np

# ----------------------------
# 1) HIER anpassen
# ----------------------------
CAM_INDEX = 2  # ggf. 1/2

# Tisch-Koordinaten (mm), Ursprung = Marker 10 (unten links, Mittelpunkt)
# X nach rechts, Y nach oben.
#
# Du trägst hier deine GEMESSENEN Werte ein.
# Tipp: Miss X entlang der unteren Kante (von ID10 nach rechts),
#      und Y entlang der linken Kante (von ID10 nach oben).
#
# IDs / Positionen laut dir:
# 10 unten links (Origin)
# 11 unten rechts
# 12 mitte links
# 13 oben links
# 14 mitte rechts
# 15 oben rechts
# 16 innen unten
# 17 innen oben
MARKER_TABLE_MM = {
    10: (0.0,   0.0),    # unten links = Ursprung

    11: (720,  0.0),    # unten rechts:  X = Tischlaenge, Y=0
    13: (0.0,   1100),   # oben links:    X=0, Y = Tischbreite
    15: (720,  1100),   # oben rechts:   X = Tischlaenge, Y = Tischbreite

    12: (0.0,   500),   # mitte links:   X=0, Y = mittlere Hoehe (messen)
    14: (720,  500),   # mitte rechts:  X = Tischlaenge, Y = mittlere Hoehe (messen)

    16: (360,  250),   # innen unten:   X,Y messen
    17: (360,  800),   # innen oben:    X,Y messen
}

# Du MUSST die 4 Eckmarker sichtbar haben:
REQUIRED_CORNER_IDS = [10, 11, 13, 15]

# Optional: zwinge, dass alle 8 Marker sichtbar sind
REQUIRED_ALL = False

# RANSAC Threshold in Pixeln (kleiner = strenger, kann aber "failen")
RANSAC_THRESH = 2.0

# ----------------------------
# ArUco Setup
# ----------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

H = None

def marker_center(corners_4x2: np.ndarray) -> np.ndarray:
    return corners_4x2.mean(axis=0)

def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt = np.array([[pixel_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]

def on_mouse(event, x, y, flags, param):
    global H
    if event == cv2.EVENT_LBUTTONDOWN and H is not None:
        table_xy = pixel_to_table_xy(np.array([x, y], dtype=np.float32), H)
        print(f"CLICK pixel=({x},{y}) -> table_mm=({table_xy[0]:.1f}, {table_xy[1]:.1f})")

def sanity_check_mapping(mapping: dict):
    missing = []
    for k, v in mapping.items():
        if v[0] is None or v[1] is None:
            missing.append(k)
    if missing:
        print("\nWARN: In MARKER_TABLE_MM sind noch None-Werte.")
        print("      Bitte mm-Werte eintragen fuer IDs:", missing)
        print("      Das Skript wird so NICHT kalibrieren.\n")
        return False
    return True

if not sanity_check_mapping(MARKER_TABLE_MM):
    print("Beende, bis alle mm-Werte gesetzt sind.")
    raise SystemExit(1)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden. CAM_INDEX prüfen (0/1/2...).")

cv2.namedWindow("view")
cv2.setMouseCallback("view", on_mouse)

print("INFO: Fenster anklicken, dann auf Punkte im Bild klicken -> Ausgabe in mm.")
print("INFO: Taste 'q' zum Beenden.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("WARN: Konnte kein Frame lesen.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    vis = frame.copy()
    H = None

    if ids is not None:
        ids_flat = ids.flatten().tolist()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        # 1) Prüfen: sind die 4 Ecken da?
        corners_present = all(mid in ids_flat for mid in REQUIRED_CORNER_IDS)

        # 2) Sammle alle sichtbaren Marker, die wir kennen
        pixel_pts = []
        table_pts = []

        for mid, (tx, ty) in MARKER_TABLE_MM.items():
            if mid not in ids_flat:
                continue
            idx = ids_flat.index(mid)
            c = corners[idx].reshape(4, 2).astype(np.float32)
            center = marker_center(c)

            pixel_pts.append(center)
            table_pts.append([tx, ty])

            cv2.circle(vis, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(
                vis, f"ID {mid}",
                tuple(center.astype(int) + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        pixel_pts = np.array(pixel_pts, dtype=np.float32)
        table_pts = np.array(table_pts, dtype=np.float32)

        enough_points = len(pixel_pts) >= 4

        if corners_present and enough_points and (not REQUIRED_ALL or len(pixel_pts) == len(MARKER_TABLE_MM)):
            H, inliers = cv2.findHomography(
                pixel_pts, table_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_THRESH
            )

            if H is not None:
                np.save("H.npy", H)
                inl = int(inliers.sum()) if inliers is not None else len(pixel_pts)
                cv2.putText(vis, f"H OK (pixel->table) points={len(pixel_pts)} inliers={inl}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Homography failed", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            msg = []
            if not corners_present:
                msg.append("need corner IDs 10,11,13,15 visible")
            if not enough_points:
                msg.append("need >=4 known markers")
            if REQUIRED_ALL and len(pixel_pts) != len(MARKER_TABLE_MM):
                msg.append("need ALL markers visible")
            cv2.putText(vis, " / ".join(msg), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        cv2.putText(vis, "No markers detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("view", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
