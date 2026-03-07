import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from threading import Lock


model_path  = 'pose_landmarker_heavy.task'

cap = cv2.VideoCapture(0)  # change if needed

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
result_lock = Lock()
NOSE_LANDMARK_INDEX = 0
SELECTED_LANDMARK_INDEXES = [11 ,12]


def draw_nose_coordinate(frame, pose_landmarks, width, height, landmark_indexes=None):
    if landmark_indexes is None:
        landmark_indexes = [NOSE_LANDMARK_INDEX]
    elif isinstance(landmark_indexes, int):
        landmark_indexes = [landmark_indexes]

    for text_row, landmark_index in enumerate(landmark_indexes):
        if landmark_index < 0 or landmark_index >= len(pose_landmarks):
            continue

        landmark = pose_landmarks[landmark_index]
        landmark_x = int(landmark.x * width)
        landmark_y = int(landmark.y * height)

        cv2.circle(frame, (landmark_x, landmark_y), 6, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"LM {landmark_index}: ({landmark_x}, {landmark_y})",
            (10, 30 + (text_row * 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, #font scale
            (0, 0, 255), #color
            2,           #thickness
            cv2.LINE_AA, #anti-aliased edge for smoother text
        )
def dist_calc(pose_landmarks, landmark_indexes):
    if landmark_indexes is None:
        print("No indexes given, exiting")
        return
    ls = pose_landmarks[landmark_indexes[0]]
    rs = pose_landmarks[landmark_indexes[1]]

    lsx, lsy = ls.x * w, ls.y *h
    rsx, rsy = rs.x * w, rs.y *h
    shoulder_dist_px = ((lsx - rsx) ** 2 + (lsy - rsy) ** 2) ** 0.5
    cv2.putText(frame, f"Shoulder distance: {shoulder_dist_px:.1f}px",
                (10,300),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,255,0),
                2,cv2.LINE_AA)


def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_poses=3,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame not detected')
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time()*1000)
        start = time.time()
        landmarker.detect_async(mp_image, timestamp)
        end = time.time()

        print("Inference time:", (end - start)*1000, "ms")
        with result_lock:
            result = latest_result

        if result and result.pose_landmarks:
            h, w, _ = frame.shape
            
            for pose_landmarks in result.pose_landmarks:
                for lm in pose_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
                draw_nose_coordinate(frame, pose_landmarks, w, h, SELECTED_LANDMARK_INDEXES)
                dist_calc(pose_landmarks, SELECTED_LANDMARK_INDEXES)

        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
