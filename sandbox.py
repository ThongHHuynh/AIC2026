import math
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from rplidar import RPLidar

# ---------------- USER SETTINGS ----------------
CAM_INDEX = 0                 # try 0,1,2...
LIDAR_PORT = "/dev/ttyUSB0"   # change if needed
LIDAR_BAUD = 460800           # C1 often uses 460800
CAM_HFOV_DEG = 62.2           # Pi Cam v2 ≈ 62.2 deg (change if different)
MIN_RANGE_M = 0.12
MAX_RANGE_M = 12.0
# ------------------------------------------------

# Shared LiDAR data: latest distance by angle-degree index [0..359]
latest_ranges_m = np.full(360, np.nan, dtype=np.float32)
ranges_lock = threading.Lock()
stop_flag = False


def lidar_thread_fn():
    """Continuously reads LiDAR scans and fills latest_ranges_m[deg]."""
    global latest_ranges_m, stop_flag
    lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=1)

    try:
        # optional: check info
        info = lidar.get_info()
        print("LiDAR info:", info)

        for scan in lidar.iter_scans(max_buf_meas=2000):
            if stop_flag:
                break

            # scan: list of (quality, angle_deg, distance_mm)
            temp = np.full(360, np.nan, dtype=np.float32)

            for (_, angle_deg, dist_mm) in scan:
                deg = int(round(angle_deg)) % 360
                dist_m = dist_mm / 1000.0

                if MIN_RANGE_M <= dist_m <= MAX_RANGE_M:
                    temp[deg] = dist_m

            # Update shared array (atomic-ish)
            with ranges_lock:
                latest_ranges_m = temp

    except Exception as e:
        print("LiDAR thread error:", e)
    finally:
        try:
            lidar.stop()
        except Exception:
            pass
        try:
            lidar.disconnect()
        except Exception:
            pass
        print("LiDAR thread stopped.")


def get_distance_at_bearing(bearing_deg: float, search_window_deg: int = 3):
    """
    Returns the closest valid LiDAR distance near bearing_deg by searching +/- window.
    """
    bearing_deg = bearing_deg % 360
    with ranges_lock:
        arr = latest_ranges_m.copy()

    best = np.nan
    best_abs = 1e9

    for d in range(-search_window_deg, search_window_deg + 1):
        idx = int(round(bearing_deg + d)) % 360
        val = arr[idx]
        if np.isfinite(val):
            if abs(d) < best_abs:
                best_abs = abs(d)
                best = val

    if np.isfinite(best):
        return float(best)
    return None


def pose_center_x(landmarks):
    """
    Returns normalized x center [0..1] using hips if possible else shoulders.
    MediaPipe Pose landmarks are indexed by PoseLandmark enum.
    """
    # indices: left/right hip = 23/24, left/right shoulder = 11/12
    lh = landmarks[23]
    rh = landmarks[24]
    if lh.visibility > 0.5 and rh.visibility > 0.5:
        return 0.5 * (lh.x + rh.x)

    ls = landmarks[11]
    rs = landmarks[12]
    return 0.5 * (ls.x + rs.x)


def main():
    global stop_flag

    # Start LiDAR reading in background thread
    t = threading.Thread(target=lidar_thread_fn, daemon=True)
    t.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        stop_flag = True
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 1 or 2.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    hfov_rad = math.radians(CAM_HFOV_DEG)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No camera frame.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            dist_m = None
            bearing_deg = None

            if res.pose_landmarks:
                # Draw skeleton
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Compute person center x -> bearing
                lm = res.pose_landmarks.landmark
                cx = pose_center_x(lm)  # 0..1

                # Map image x to bearing:
                # cx=0.5 -> 0 deg (forward), left negative, right positive
                bearing_rad = (cx - 0.5) * hfov_rad
                bearing_deg = math.degrees(bearing_rad)

                # LiDAR convention here:
                # We assume LiDAR 0 deg = forward.
                # If your LiDAR is rotated, add an offset like: bearing_deg += OFFSET_DEG
                dist_m = get_distance_at_bearing(bearing_deg)

                # Visualize center line
                px = int(cx * w)
                cv2.line(frame, (px, 0), (px, h), (255, 255, 255), 2)

            # Overlay text
            if bearing_deg is not None:
                cv2.putText(frame, f"Bearing ~ {bearing_deg:+.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if dist_m is not None:
                cv2.putText(frame, f"LiDAR distance ~ {dist_m:.2f} m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "LiDAR distance: (no valid reading)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Pose + LiDAR distance (pure python)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)  # let lidar thread exit


if __name__ == "__main__":
    main()