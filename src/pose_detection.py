import cv2
import mediapipe as mp
import time

CAMERA_INDEX = 0
MODEL_PATH = "models/pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    (9, 10),

    (11, 12),

    (11, 13), (13, 15),
    (12, 14), (14, 16),

    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),

    (11, 23),
    (12, 24),
    (23, 24),

    (23, 25), (25, 27),
    (24, 26), (26, 28),

    (27, 29), (29, 31),
    (28, 30), (30, 32)
]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("No se pudo abrir la camara.")
    exit()

print("Pose Challenge AI iniciado.")
print("Ponte frente a la camara.")
print("Presiona Q para cerrar.")

with PoseLandmarker.create_from_options(options) as landmarker:

    start_time = time.perf_counter()
    last_timestamp = -1

    while True:

        ret, frame = cap.read()

        if not ret:
            print("No se pudo obtener imagen.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1

        last_timestamp = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            height, width, _ = frame.shape

            for connection in POSE_CONNECTIONS:

                start_idx = connection[0]
                end_idx = connection[1]

                start = landmarks[start_idx]
                end = landmarks[end_idx]

                x1 = int(start.x * width)
                y1 = int(start.y * height)

                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            for i, landmark in enumerate(landmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )

            cv2.putText(
                frame,
                "CUERPO DETECTADO",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO SE DETECTA EL CUERPO",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Pose Challenge AI - MediaPipe",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()