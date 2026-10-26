import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter


# ==========================================
# CONFIGURACION
# ==========================================

CAMERA_INDEX = 0

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
CLASSIFIER_PATH = "models/pose_classifier.pkl"

# Bajamos el umbral para que no salga
# "INCIERTO" tan facilmente
MIN_CONFIDENCE = 0.25

# Historial para estabilizar la prediccion
prediction_history = deque(maxlen=10)


# ==========================================
# CONEXIONES DEL CUERPO
# ==========================================

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

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32)
]


# ==========================================
# NORMALIZACION
# Debe ser IGUAL a la usada al entrenar
# ==========================================

def normalize_landmarks(landmarks):

    points = np.array([
        [
            landmark.x,
            landmark.y,
            landmark.z
        ]
        for landmark in landmarks
    ], dtype=np.float32)

    # Centro de caderas
    hip_center = (
        points[23] +
        points[24]
    ) / 2.0

    points = points - hip_center

    # Centro de hombros
    shoulder_center = (
        points[11] +
        points[12]
    ) / 2.0

    hip_center_after = (
        points[23] +
        points[24]
    ) / 2.0

    torso_length = np.linalg.norm(
        shoulder_center[:2]
        -
        hip_center_after[:2]
    )

    if torso_length < 0.0001:
        return None

    points = points / torso_length

    return points.flatten()


# ==========================================
# CARGAR MODELO ENTRENADO
# ==========================================

print("Cargando modelo entrenado...")

model_data = joblib.load(
    CLASSIFIER_PATH
)

classifier = model_data["model"]

print("Modelo cargado correctamente.")

print("\nClases disponibles:")

for label in classifier.classes_:
    print(" -", label)


# ==========================================
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ==========================================
# CAMARA
# ==========================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not cap.isOpened():

    print("No se pudo abrir la camara.")
    exit()


print("\n======================================")
print("POSE CHALLENGE AI")
print("======================================")

print("\nPonte frente a la camara.")
print("Debe verse todo tu cuerpo.")
print("Presiona Q para cerrar.")


# ==========================================
# DETECCION
# ==========================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    start_time = time.perf_counter()

    last_timestamp = -1

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "No se pudo obtener imagen."
            )

            break


        # Espejo para que sea mas natural
        frame = cv2.flip(frame, 1)


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )


        timestamp_ms = int(
            (
                time.perf_counter()
                -
                start_time
            )
            * 1000
        )


        if timestamp_ms <= last_timestamp:

            timestamp_ms = (
                last_timestamp + 1
            )


        last_timestamp = timestamp_ms


        result = landmarker.detect_for_video(

            mp_image,

            timestamp_ms
        )


        # ==================================
        # SI HAY CUERPO
        # ==================================

        if result.pose_landmarks:

            landmarks = (
                result.pose_landmarks[0]
            )


            height, width, _ = frame.shape


            # ==============================
            # DIBUJAR ESQUELETO
            # ==============================

            for connection in POSE_CONNECTIONS:

                start = landmarks[
                    connection[0]
                ]

                end = landmarks[
                    connection[1]
                ]


                x1 = int(
                    start.x * width
                )

                y1 = int(
                    start.y * height
                )

                x2 = int(
                    end.x * width
                )

                y2 = int(
                    end.y * height
                )


                cv2.line(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 255, 255),

                    2
                )


            # ==============================
            # DIBUJAR PUNTOS
            # ==============================

            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(

                    frame,

                    (x, y),

                    4,

                    (0, 255, 0),

                    -1
                )


            # ==============================
            # PREPARAR FEATURES
            # ==============================

            features = normalize_landmarks(
                landmarks
            )


            if features is not None:

                features = features.reshape(
                    1,
                    -1
                )


                # ==========================
                # PREDICCION
                # ==========================

                prediction = classifier.predict(
                    features
                )[0]


                probabilities = (
                    classifier.predict_proba(
                        features
                    )[0]
                )


                confidence = float(
                    np.max(probabilities)
                )


                # ==========================
                # TOP 3 PREDICCIONES
                # ==========================

                best_indices = (
                    np.argsort(
                        probabilities
                    )[::-1][:3]
                )


                top_predictions = []

                for idx in best_indices:

                    pose_name = (
                        classifier.classes_[idx]
                    )

                    pose_probability = (
                        probabilities[idx]
                        * 100
                    )

                    top_predictions.append(
                        (
                            pose_name,
                            pose_probability
                        )
                    )


                # ==========================
                # SUAVIZADO
                # ==========================

                if confidence >= MIN_CONFIDENCE:

                    prediction_history.append(
                        prediction
                    )


                    most_common = Counter(
                        prediction_history
                    ).most_common(1)


                    if most_common:

                        stable_prediction = (
                            most_common[0][0]
                        )

                    else:

                        stable_prediction = (
                            prediction
                        )

                else:

                    stable_prediction = (
                        "INCIERTO"
                    )


                # ==========================
                # FONDO DEL PANEL
                # ==========================

                cv2.rectangle(
                    frame,
                    (20, 20),
                    (620, 280),
                    (0, 0, 0),
                    -1
                )


                # ==========================
                # TITULO
                # ==========================

                cv2.putText(

                    frame,

                    "POSE CHALLENGE AI",

                    (35, 55),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    (255, 255, 255),

                    2
                )


                # ==========================
                # POSTURA
                # ==========================

                cv2.putText(

                    frame,

                    f"POSTURA: {stable_prediction.upper()}",

                    (35, 100),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1,

                    (0, 255, 0),

                    2
                )


                # ==========================
                # CONFIANZA
                # ==========================

                cv2.putText(

                    frame,

                    f"CONFIANZA: {confidence * 100:.1f}%",

                    (35, 140),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 255),

                    2
                )


                # ==========================
                # TOP 3
                # ==========================

                y_text = 180

                for i, (
                    pose_name,
                    pose_probability
                ) in enumerate(
                    top_predictions,
                    start=1
                ):

                    text = (
                        f"{i}. "
                        f"{pose_name.upper()} "
                        f"{pose_probability:.1f}%"
                    )


                    cv2.putText(

                        frame,

                        text,

                        (35, y_text),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (255, 255, 255),

                        2
                    )


                    y_text += 35


        else:

            prediction_history.clear()


            cv2.rectangle(
                frame,
                (20, 20),
                (600, 90),
                (0, 0, 0),
                -1
            )


            cv2.putText(

                frame,

                "NO SE DETECTA EL CUERPO",

                (35, 65),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 0, 255),

                2
            )


        # ==================================
        # MOSTRAR CAMARA
        # ==================================

        cv2.imshow(

            "Pose Challenge AI - Clasificador",

            frame
        )


        # ==================================
        # SALIR CON Q
        # ==================================

        if (
            cv2.waitKey(1)
            & 0xFF
            ==
            ord("q")
        ):

            break


# ==========================================
# CERRAR
# ==========================================

cap.release()

cv2.destroyAllWindows()