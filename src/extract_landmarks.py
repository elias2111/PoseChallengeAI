import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from pathlib import Path
from collections import Counter


# ==========================================
# CONFIGURACION
# ==========================================

DATASET_DIR = Path("dataset/raw")

MODEL_PATH = "models/pose_landmarker_lite.task"

OUTPUT_FILE = Path("dataset/landmarks.csv")


# Nombres que podemos encontrar dentro
# de las carpetas del dataset
POSE_ALIASES = {

    "downdog": [
        "downdog",
        "downwarddog",
        "downwardfacingdog"
    ],

    "goddess": [
        "goddess",
        "godess"
    ],

    "plank": [
        "plank"
    ],

    "tree": [
        "tree"
    ],

    "warrior2": [
        "warrior2",
        "warrioriipose",
        "warriorii"
    ]
}


# ==========================================
# FUNCIONES
# ==========================================

def clean_name(text):

    text = text.lower()

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def detect_label(path):

    complete_path = clean_name(
        str(path)
    )

    for label, aliases in POSE_ALIASES.items():

        for alias in aliases:

            if clean_name(alias) in complete_path:
                return label

    return None


def detect_split(path):

    parts = [
        clean_name(part)
        for part in path.parts
    ]

    if "train" in parts or "training" in parts:
        return "train"

    if "test" in parts or "testing" in parts:
        return "test"

    return "unknown"


def normalize_landmarks(landmarks):

    points = np.array([
        [
            landmark.x,
            landmark.y,
            landmark.z
        ]
        for landmark in landmarks
    ], dtype=np.float32)

    # --------------------------------------
    # Centro del cuerpo = mitad de caderas
    # MediaPipe:
    # 23 = cadera izquierda
    # 24 = cadera derecha
    # --------------------------------------

    hip_center = (
        points[23] +
        points[24]
    ) / 2.0

    points = points - hip_center

    # --------------------------------------
    # Escala usando torso
    # 11/12 = hombros
    # 23/24 = caderas
    # --------------------------------------

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
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5
)


# ==========================================
# BUSCAR IMAGENES
# ==========================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


images = [

    path

    for path in DATASET_DIR.rglob("*")

    if (
        path.is_file()
        and path.suffix.lower() in extensions
    )
]


print("======================================")
print("POSE CHALLENGE AI")
print("EXTRACCION DE LANDMARKS")
print("======================================")

print(
    f"\nImagenes encontradas: {len(images)}"
)


if len(images) == 0:

    print(
        "\nERROR: No se encontraron imagenes."
    )

    print(
        "Revisa dataset/raw"
    )

    exit()


# ==========================================
# COLUMNAS CSV
# ==========================================

columns = []

for i in range(33):

    columns.append(f"x{i}")
    columns.append(f"y{i}")
    columns.append(f"z{i}")

columns += [
    "label",
    "split",
    "image"
]


data = []

skipped = 0

label_counter = Counter()


# ==========================================
# PROCESAMIENTO
# ==========================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    for number, image_path in enumerate(
        images,
        start=1
    ):

        label = detect_label(
            image_path
        )

        if label is None:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            skipped += 1
            continue

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_image
        )

        result = landmarker.detect(
            mp_image
        )

        if not result.pose_landmarks:

            skipped += 1
            continue

        landmarks = (
            result.pose_landmarks[0]
        )

        features = normalize_landmarks(
            landmarks
        )

        if features is None:

            skipped += 1
            continue

        split = detect_split(
            image_path
        )

        row = (
            list(features)
            +
            [
                label,
                split,
                str(image_path)
            ]
        )

        data.append(
            row
        )

        label_counter[label] += 1

        if number % 50 == 0:

            print(
                f"Procesando "
                f"{number}/{len(images)}"
            )


# ==========================================
# GUARDAR CSV
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.DataFrame(
    data,
    columns=columns
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n======================================")
print("EXTRACCION TERMINADA")
print("======================================")

print(
    f"Landmarks obtenidos: {len(df)}"
)

print(
    f"Imagenes descartadas: {skipped}"
)

print("\nMuestras por postura:")

for label, count in sorted(
    label_counter.items()
):

    print(
        f"{label:12} -> {count}"
    )

print(
    f"\nArchivo generado:"
)

print(
    OUTPUT_FILE
)