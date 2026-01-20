import cv2
import numpy as np

# ----------------------------
# 1) HIER anpassen
# ----------------------------
CAM_INDEX = 0  # evtl. 1, wenn du mehrere Kameras hast

# Deine 4 Marker-IDs (in fester Reihenfolge!)
MARKER_IDS = [10, 11, 12, 13]

# Realwelt-Koordinaten (mm) passend zur obigen Reihenfolge
# Beispiel: Rechteck 600mm x 400mm
TABLE_POINTS_MM = np.array([
    [0.0,   0.0],    # ID 10
    [860.0, 0.0],    # ID 11
    [860.0, 490.0],  # ID 12
    [0.0,   490.0],  # ID 13
], dtype=np.float32)

# ----------------------------
# ArUco Setup
# ----------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ARUCO_PARAMS.minMarkerPerimeterRate = 0.02
ARUCO_PARAMS.maxMarkerPerimeterRate = 4.0
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

H = None

def marker_center(corners_4x2: np.ndarray) -> np.ndarray:
    return corners_4x2.mean(axis=0)

def pixel_to_table_xy(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt = np.array([[pixel_xy]], dtype=np.float32)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)
    return out[0, 0]

def on_mouse(event, x, y, flags, param):
    global H
    if event == cv2.EVENT_LBUTTONDOWN and H is not None:
        table_xy = pixel_to_table_xy(np.array([x, y], dtype=np.float32), H)
        print(f"CLICK pixel=({x},{y}) -> table_mm=({table_xy[0]:.1f}, {table_xy[1]:.1f})")

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
    corners, ids, rejected = detector.detectMarkers(gray)

    vis = frame.copy()
    H = None

    if ids is not None:
        ids_flat = ids.flatten().tolist()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        pixel_points = []
        found_all = True

        for mid in MARKER_IDS:
            if mid not in ids_flat:
                found_all = False
                break

            idx = ids_flat.index(mid)
            c = corners[idx].reshape(4, 2).astype(np.float32)
            center = marker_center(c)
            pixel_points.append(center)

            cv2.circle(vis, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(vis, f"ID {mid}", tuple(center.astype(int) + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if found_all:
            pixel_points = np.array(pixel_points, dtype=np.float32)
            H, inliers = cv2.findHomography(pixel_points, TABLE_POINTS_MM, method=0)
            cv2.putText(vis, "H OK (pixel->table)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Need all 4 markers visible", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(vis, "No markers detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("view", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
