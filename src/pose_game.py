import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import random
import time
import sys
import sklearn

from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from collections import deque, Counter


# ============================================================
# RUTAS COMPATIBLES CON PYINSTALLER
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


POSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pose_landmarker_lite.task"
)

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "pose_classifier.pkl"
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

CAMERA_INDEX = 0

MIN_CONFIDENCE = 0.25

ROUND_TIME = 15
HOLD_TIME = 0.5
TOTAL_ROUNDS = 10

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760

CAMERA_WIDTH = 760
CAMERA_HEIGHT = 428


POSES = [
    "downdog",
    "goddess",
    "plank",
    "tree",
    "warrior2"
]


POSE_NAMES = {
    "downdog": "DOWN DOG",
    "goddess": "GODDESS",
    "plank": "PLANK",
    "tree": "TREE",
    "warrior2": "WARRIOR 2"
}


POSE_INSTRUCTIONS = {
    "downdog": "Forma una V invertida",
    "goddess": "Piernas abiertas y rodillas flexionadas",
    "plank": "Mantente recto en posicion de plancha",
    "tree": "Apoya un pie sobre la otra pierna",
    "warrior2": "Brazos rectos y piernas abiertas"
}


# ============================================================
# COLORES
# ============================================================

BACKGROUND = (15, 18, 24)

PANEL = (27, 31, 40)

PANEL_LIGHT = (37, 42, 54)

WHITE = (245, 245, 245)

GRAY = (165, 170, 180)

GREEN = (46, 204, 113)

DARK_GREEN = (30, 140, 80)

YELLOW = (245, 200, 66)

RED = (231, 76, 60)


prediction_history = deque(
    maxlen=8
)


# ============================================================
# CONEXIONES DEL CUERPO
# ============================================================

POSE_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    (9, 10),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (15, 17),
    (15, 19),
    (15, 21),

    (16, 18),
    (16, 20),
    (16, 22),

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


# ============================================================
# NORMALIZAR LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    points = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z
            ]
            for landmark in landmarks
        ],
        dtype=np.float32
    )


    hip_center = (
        points[23]
        +
        points[24]
    ) / 2.0


    points = (
        points
        -
        hip_center
    )


    shoulder_center = (
        points[11]
        +
        points[12]
    ) / 2.0


    hip_center_after = (
        points[23]
        +
        points[24]
    ) / 2.0


    torso_length = np.linalg.norm(

        shoulder_center[:2]
        -
        hip_center_after[:2]

    )


    if torso_length < 0.0001:

        return None


    points = (
        points
        /
        torso_length
    )


    return points.flatten()


# ============================================================
# CREAR SECUENCIA DE POSTURAS
# ============================================================

def create_pose_sequence():

    first_group = POSES.copy()

    second_group = POSES.copy()


    random.shuffle(
        first_group
    )

    random.shuffle(
        second_group
    )


    # Evitar que termine una pose
    # y empiece inmediatamente la misma
    while (
        first_group[-1]
        ==
        second_group[0]
    ):

        random.shuffle(
            second_group
        )


    return (
        first_group
        +
        second_group
    )


# ============================================================
# FUNCIONES DE INTERFAZ
# ============================================================

def draw_text(
    surface,
    text,
    font,
    color,
    x,
    y,
    center=False
):

    rendered = font.render(
        text,
        True,
        color
    )


    rect = rendered.get_rect()


    if center:

        rect.center = (
            x,
            y
        )

    else:

        rect.topleft = (
            x,
            y
        )


    surface.blit(
        rendered,
        rect
    )


    return rect


def draw_rounded_panel(
    surface,
    rect,
    color,
    radius=18
):

    pygame.draw.rect(
        surface,
        color,
        rect,
        border_radius=radius
    )


def draw_progress_bar(
    surface,
    x,
    y,
    width,
    height,
    progress,
    color
):

    progress = max(
        0,
        min(
            1,
            progress
        )
    )


    pygame.draw.rect(
        surface,
        PANEL_LIGHT,
        (
            x,
            y,
            width,
            height
        ),
        border_radius=height // 2
    )


    fill_width = int(
        width
        *
        progress
    )


    if fill_width > 0:

        pygame.draw.rect(
            surface,
            color,
            (
                x,
                y,
                fill_width,
                height
            ),
            border_radius=height // 2
        )


# ============================================================
# DIBUJAR FIGURA HUMANA
# ============================================================

def line(
    surface,
    p1,
    p2,
    color=WHITE,
    width=6
):

    pygame.draw.line(
        surface,
        color,
        p1,
        p2,
        width
    )


def draw_pose_reference(
    surface,
    pose,
    area
):

    x, y, w, h = area

    cx = (
        x
        +
        w // 2
    )

    cy = (
        y
        +
        h // 2
    )

    color = WHITE


    # ========================================================
    # TREE
    # ========================================================

    if pose == "tree":

        head = (
            cx,
            cy - 80
        )

        neck = (
            cx,
            cy - 52
        )

        hip = (
            cx,
            cy + 15
        )

        left_hand = (
            cx - 35,
            cy - 115
        )

        right_hand = (
            cx + 35,
            cy - 115
        )

        left_elbow = (
            cx - 30,
            cy - 75
        )

        right_elbow = (
            cx + 30,
            cy - 75
        )

        left_knee = (
            cx,
            cy + 80
        )

        left_foot = (
            cx,
            cy + 125
        )

        right_knee = (
            cx + 42,
            cy + 48
        )

        right_foot = (
            cx + 5,
            cy + 72
        )


        pygame.draw.circle(
            surface,
            color,
            head,
            18,
            4
        )


        line(
            surface,
            neck,
            hip
        )

        line(
            surface,
            neck,
            left_elbow
        )

        line(
            surface,
            left_elbow,
            left_hand
        )

        line(
            surface,
            neck,
            right_elbow
        )

        line(
            surface,
            right_elbow,
            right_hand
        )

        line(
            surface,
            hip,
            left_knee
        )

        line(
            surface,
            left_knee,
            left_foot
        )

        line(
            surface,
            hip,
            right_knee
        )

        line(
            surface,
            right_knee,
            right_foot
        )


    # ========================================================
    # WARRIOR 2
    # ========================================================

    elif pose == "warrior2":

        head = (
            cx,
            cy - 75
        )

        shoulder = (
            cx,
            cy - 45
        )

        hip = (
            cx,
            cy + 15
        )

        left_hand = (
            cx - 100,
            cy - 45
        )

        right_hand = (
            cx + 100,
            cy - 45
        )

        left_knee = (
            cx - 55,
            cy + 65
        )

        right_knee = (
            cx + 55,
            cy + 55
        )

        left_foot = (
            cx - 105,
            cy + 115
        )

        right_foot = (
            cx + 115,
            cy + 105
        )


        pygame.draw.circle(
            surface,
            color,
            head,
            18,
            4
        )


        line(
            surface,
            shoulder,
            hip
        )

        line(
            surface,
            shoulder,
            left_hand
        )

        line(
            surface,
            shoulder,
            right_hand
        )

        line(
            surface,
            hip,
            left_knee
        )

        line(
            surface,
            left_knee,
            left_foot
        )

        line(
            surface,
            hip,
            right_knee
        )

        line(
            surface,
            right_knee,
            right_foot
        )


    # ========================================================
    # GODDESS
    # ========================================================

    elif pose == "goddess":

        head = (
            cx,
            cy - 75
        )

        shoulder = (
            cx,
            cy - 45
        )

        hip = (
            cx,
            cy + 15
        )

        left_elbow = (
            cx - 55,
            cy - 65
        )

        right_elbow = (
            cx + 55,
            cy - 65
        )

        left_hand = (
            cx - 70,
            cy - 110
        )

        right_hand = (
            cx + 70,
            cy - 110
        )

        left_knee = (
            cx - 65,
            cy + 65
        )

        right_knee = (
            cx + 65,
            cy + 65
        )

        left_foot = (
            cx - 110,
            cy + 110
        )

        right_foot = (
            cx + 110,
            cy + 110
        )


        pygame.draw.circle(
            surface,
            color,
            head,
            18,
            4
        )


        line(
            surface,
            shoulder,
            hip
        )

        line(
            surface,
            shoulder,
            left_elbow
        )

        line(
            surface,
            left_elbow,
            left_hand
        )

        line(
            surface,
            shoulder,
            right_elbow
        )

        line(
            surface,
            right_elbow,
            right_hand
        )

        line(
            surface,
            hip,
            left_knee
        )

        line(
            surface,
            left_knee,
            left_foot
        )

        line(
            surface,
            hip,
            right_knee
        )

        line(
            surface,
            right_knee,
            right_foot
        )


    # ========================================================
    # PLANK
    # ========================================================

    elif pose == "plank":

        head = (
            cx - 100,
            cy - 25
        )

        shoulder = (
            cx - 65,
            cy
        )

        hip = (
            cx + 15,
            cy + 10
        )

        foot = (
            cx + 115,
            cy + 25
        )

        hand = (
            cx - 90,
            cy + 85
        )


        pygame.draw.circle(
            surface,
            color,
            head,
            15,
            4
        )


        line(
            surface,
            shoulder,
            hip
        )

        line(
            surface,
            hip,
            foot
        )

        line(
            surface,
            shoulder,
            hand
        )


    # ========================================================
    # DOWN DOG
    # ========================================================

    elif pose == "downdog":

        hip = (
            cx,
            cy - 75
        )

        shoulder = (
            cx - 55,
            cy - 10
        )

        hand = (
            cx - 110,
            cy + 90
        )

        knee = (
            cx + 55,
            cy + 10
        )

        foot = (
            cx + 110,
            cy + 90
        )

        head = (
            cx - 55,
            cy + 5
        )


        pygame.draw.circle(
            surface,
            color,
            head,
            14,
            4
        )


        line(
            surface,
            hip,
            shoulder
        )

        line(
            surface,
            shoulder,
            hand
        )

        line(
            surface,
            hip,
            knee
        )

        line(
            surface,
            knee,
            foot
        )


# ============================================================
# CARGAR MODELO ENTRENADO
# ============================================================

print(
    "Cargando modelo..."
)


try:

    model_data = joblib.load(
        CLASSIFIER_PATH
    )

except Exception as error:

    print(
        "ERROR AL CARGAR EL MODELO:"
    )

    print(
        error
    )

    input(
        "Presiona ENTER para cerrar..."
    )

    sys.exit()


classifier = (
    model_data["model"]
)


print(
    "Modelo cargado correctamente."
)


# ============================================================
# CONFIGURAR MEDIAPIPE
# ============================================================

BaseOptions = (
    mp.tasks.BaseOptions
)

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

        model_asset_path=str(
            POSE_MODEL_PATH
        )

    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5

)


# ============================================================
# ABRIR CAMARA
# ============================================================

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

    print(
        "No se pudo abrir la camara."
    )

    input(
        "Presiona ENTER para cerrar..."
    )

    sys.exit()


# ============================================================
# INICIAR PYGAME
# ============================================================

pygame.init()


screen = pygame.display.set_mode(
    (
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )
)


pygame.display.set_caption(
    "Pose Challenge AI"
)


clock = pygame.time.Clock()


# ============================================================
# FUENTES
# ============================================================

font_logo = pygame.font.Font(
    None,
    56
)

font_title = pygame.font.Font(
    None,
    42
)

font_large = pygame.font.Font(
    None,
    38
)

font_medium = pygame.font.Font(
    None,
    28
)

font_small = pygame.font.Font(
    None,
    22
)


# ============================================================
# ESTADO DEL JUEGO
# ============================================================

game_state = "START"


pose_sequence = (
    create_pose_sequence()
)


round_number = 1


target_pose = (
    pose_sequence[0]
)


score = 0


correct_poses = 0


round_start = 0


match_start = None


success_message_start = None


# ============================================================
# REINICIAR JUEGO
# ============================================================

def reset_game():

    global pose_sequence
    global round_number
    global target_pose
    global score
    global correct_poses
    global round_start
    global match_start
    global success_message_start
    global game_state


    pose_sequence = (
        create_pose_sequence()
    )


    round_number = 1


    target_pose = (
        pose_sequence[0]
    )


    score = 0


    correct_poses = 0


    round_start = (
        time.time()
    )


    match_start = None


    success_message_start = None


    prediction_history.clear()


    game_state = "PLAYING"


# ============================================================
# TEMPORIZACION MEDIAPIPE
# ============================================================

start_time = (
    time.perf_counter()
)


last_timestamp = -1


# ============================================================
# LOOP PRINCIPAL
# ============================================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    running = True


    while running:

        # ====================================================
        # EVENTOS
        # ====================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False


            if event.type == pygame.KEYDOWN:

                if event.key in (
                    pygame.K_q,
                    pygame.K_ESCAPE
                ):

                    running = False


                if (
                    game_state == "START"
                    and
                    event.key == pygame.K_RETURN
                ):

                    reset_game()


                if (
                    game_state == "GAME_OVER"
                    and
                    event.key == pygame.K_r
                ):

                    reset_game()


        # ====================================================
        # LEER CAMARA
        # ====================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "No se pudo obtener imagen."
            )

            break


        # Modo espejo
        frame = cv2.flip(
            frame,
            1
        )


        stable_prediction = (
            "NINGUNA"
        )


        confidence = 0.0


        # ====================================================
        # PREPARAR FRAME PARA MEDIAPIPE
        # ====================================================

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
            *
            1000
        )


        if (
            timestamp_ms
            <=
            last_timestamp
        ):

            timestamp_ms = (
                last_timestamp
                +
                1
            )


        last_timestamp = (
            timestamp_ms
        )


        result = (
            landmarker.detect_for_video(

                mp_image,

                timestamp_ms

            )
        )


        # ====================================================
        # SI DETECTA CUERPO
        # ====================================================

        if result.pose_landmarks:

            landmarks = (
                result.pose_landmarks[0]
            )


            height, width, _ = (
                frame.shape
            )


            # =================================================
            # DIBUJAR ESQUELETO
            # =================================================

            for connection in POSE_CONNECTIONS:

                a = landmarks[
                    connection[0]
                ]

                b = landmarks[
                    connection[1]
                ]


                p1 = (
                    int(
                        a.x
                        *
                        width
                    ),

                    int(
                        a.y
                        *
                        height
                    )
                )


                p2 = (
                    int(
                        b.x
                        *
                        width
                    ),

                    int(
                        b.y
                        *
                        height
                    )
                )


                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 255, 255),
                    2
                )


            # =================================================
            # DIBUJAR PUNTOS
            # =================================================

            for landmark in landmarks:

                x = int(
                    landmark.x
                    *
                    width
                )


                y = int(
                    landmark.y
                    *
                    height
                )


                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # CLASIFICACION
            # =================================================

            features = (
                normalize_landmarks(
                    landmarks
                )
            )


            if features is not None:

                features = (
                    features.reshape(
                        1,
                        -1
                    )
                )


                prediction = (
                    classifier.predict(
                        features
                    )[0]
                )


                probabilities = (
                    classifier.predict_proba(
                        features
                    )[0]
                )


                confidence = float(
                    np.max(
                        probabilities
                    )
                )


                if (
                    confidence
                    >=
                    MIN_CONFIDENCE
                ):

                    prediction_history.append(
                        prediction
                    )


                    stable_prediction = (
                        Counter(
                            prediction_history
                        )
                        .most_common(1)[0][0]
                    )

                else:

                    stable_prediction = (
                        "INCIERTO"
                    )


        else:

            prediction_history.clear()


            stable_prediction = (
                "SIN CUERPO"
            )


        # ====================================================
        # LOGICA DEL JUEGO
        # ====================================================

        time_left = ROUND_TIME


        hold_progress = 0


        if game_state == "PLAYING":

            elapsed = (
                time.time()
                -
                round_start
            )


            time_left = max(
                0,
                ROUND_TIME
                -
                elapsed
            )


            # =================================================
            # POSTURA CORRECTA
            # =================================================

            if (
                stable_prediction
                ==
                target_pose
                and
                confidence
                >=
                MIN_CONFIDENCE
            ):

                if match_start is None:

                    match_start = (
                        time.time()
                    )


                hold_elapsed = (
                    time.time()
                    -
                    match_start
                )


                hold_progress = min(
                    1,
                    hold_elapsed
                    /
                    HOLD_TIME
                )


                if (
                    hold_elapsed
                    >=
                    HOLD_TIME
                ):

                    score += 100


                    correct_poses += 1


                    success_message_start = (
                        time.time()
                    )


                    round_number += 1


                    prediction_history.clear()


                    match_start = None


                    if (
                        round_number
                        >
                        TOTAL_ROUNDS
                    ):

                        game_state = (
                            "GAME_OVER"
                        )


                    else:

                        target_pose = (
                            pose_sequence[
                                round_number
                                -
                                1
                            ]
                        )


                        round_start = (
                            time.time()
                        )


            else:

                match_start = None

                hold_progress = 0


            # =================================================
            # TIEMPO AGOTADO
            # =================================================

            if (
                time_left <= 0
                and
                game_state
                ==
                "PLAYING"
            ):

                round_number += 1


                prediction_history.clear()


                match_start = None


                if (
                    round_number
                    >
                    TOTAL_ROUNDS
                ):

                    game_state = (
                        "GAME_OVER"
                    )


                else:

                    target_pose = (
                        pose_sequence[
                            round_number
                            -
                            1
                        ]
                    )


                    round_start = (
                        time.time()
                    )


        # ====================================================
        # CONVERTIR CAMARA A PYGAME
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frame_surface = (
            pygame.image.frombuffer(

                frame_rgb.tobytes(),

                (
                    frame_rgb.shape[1],
                    frame_rgb.shape[0]
                ),

                "RGB"

            ).copy()
        )


        frame_surface = (
            pygame.transform.smoothscale(

                frame_surface,

                (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT
                )

            )
        )


        # ====================================================
        # FONDO
        # ====================================================

        screen.fill(
            BACKGROUND
        )


        # ====================================================
        # TITULO
        # ====================================================

        draw_text(
            screen,
            "POSE CHALLENGE",
            font_logo,
            WHITE,
            35,
            22
        )


        draw_text(
            screen,
            "AI",
            font_logo,
            GREEN,
            400,
            22
        )


        # ====================================================
        # AREA DE CAMARA
        # ====================================================

        camera_rect = pygame.Rect(
            30,
            95,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )


        draw_rounded_panel(
            screen,
            camera_rect,
            PANEL
        )


        screen.blit(
            frame_surface,
            (
                30,
                95
            )
        )


        pygame.draw.rect(
            screen,
            PANEL_LIGHT,
            camera_rect,
            3,
            border_radius=15
        )


        # ====================================================
        # PANTALLA INICIAL
        # ====================================================

        if game_state == "START":

            overlay = pygame.Surface(
                (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT
                ),
                pygame.SRCALPHA
            )


            overlay.fill(
                (
                    0,
                    0,
                    0,
                    185
                )
            )


            screen.blit(
                overlay,
                (
                    30,
                    95
                )
            )


            draw_text(
                screen,
                "POSE CHALLENGE AI",
                font_title,
                WHITE,
                410,
                210,
                True
            )


            draw_text(
                screen,
                "Imita las posturas antes de que termine el tiempo",
                font_medium,
                GRAY,
                410,
                270,
                True
            )


            draw_text(
                screen,
                "5 posturas - 10 rondas - 100 puntos por acierto",
                font_small,
                GRAY,
                410,
                310,
                True
            )


            draw_text(
                screen,
                "ENTER PARA COMENZAR",
                font_large,
                GREEN,
                410,
                385,
                True
            )


        # ====================================================
        # PANEL DERECHO
        # ====================================================

        panel_rect = pygame.Rect(
            820,
            95,
            430,
            570
        )


        draw_rounded_panel(
            screen,
            panel_rect,
            PANEL
        )


        if game_state == "PLAYING":

            draw_text(
                screen,
                "POSTURA OBJETIVO",
                font_small,
                GRAY,
                845,
                118
            )


            draw_text(
                screen,
                POSE_NAMES[
                    target_pose
                ],
                font_title,
                GREEN,
                845,
                148
            )


            draw_text(
                screen,
                POSE_INSTRUCTIONS[
                    target_pose
                ],
                font_small,
                WHITE,
                845,
                195
            )


            # =================================================
            # REFERENCIA VISUAL
            # =================================================

            pose_area = (
                845,
                220,
                380,
                225
            )


            pygame.draw.rect(
                screen,
                PANEL_LIGHT,
                pose_area,
                border_radius=15
            )


            draw_pose_reference(
                screen,
                target_pose,
                pose_area
            )


            # =================================================
            # MODELO
            # =================================================

            draw_text(
                screen,
                "MODELO DETECTA",
                font_small,
                GRAY,
                845,
                462
            )


            if (
                stable_prediction
                ==
                target_pose
            ):

                detection_color = GREEN

            else:

                detection_color = YELLOW


            display_prediction = (
                POSE_NAMES.get(
                    stable_prediction,
                    stable_prediction
                )
            )


            draw_text(
                screen,
                display_prediction,
                font_large,
                detection_color,
                845,
                490
            )


            draw_text(
                screen,
                f"Confianza: {confidence * 100:.1f}%",
                font_small,
                WHITE,
                845,
                535
            )


            # =================================================
            # BARRA MANTENER
            # =================================================

            draw_text(
                screen,
                "Mantener postura",
                font_small,
                GRAY,
                845,
                570
            )


            draw_progress_bar(
                screen,
                845,
                600,
                370,
                14,
                hold_progress,
                GREEN
            )


        # ====================================================
        # TEMPORIZADOR
        # ====================================================

        if game_state == "PLAYING":

            time_ratio = (
                time_left
                /
                ROUND_TIME
            )


            timer_color = GREEN


            if time_left <= 7:

                timer_color = YELLOW


            if time_left <= 3:

                timer_color = RED


            draw_text(
                screen,
                f"TIEMPO {time_left:04.1f}s",
                font_medium,
                timer_color,
                35,
                550
            )


            draw_progress_bar(
                screen,
                35,
                585,
                755,
                16,
                time_ratio,
                timer_color
            )


        # ====================================================
        # ESTADISTICAS
        # ====================================================

        stats_y = 625


        draw_text(
            screen,
            f"PUNTOS  {score}",
            font_large,
            WHITE,
            35,
            stats_y
        )


        draw_text(
            screen,
            f"CORRECTAS  {correct_poses}/{TOTAL_ROUNDS}",
            font_medium,
            GRAY,
            275,
            stats_y + 5
        )


        if game_state == "PLAYING":

            draw_text(
                screen,
                f"RONDA  {round_number}/{TOTAL_ROUNDS}",
                font_medium,
                GRAY,
                570,
                stats_y + 5
            )


        # ====================================================
        # MENSAJE DE ACIERTO
        # ====================================================

        if (
            success_message_start
            is not None
        ):

            success_elapsed = (
                time.time()
                -
                success_message_start
            )


            if success_elapsed < 1.2:

                success_box = pygame.Rect(
                    230,
                    260,
                    360,
                    100
                )


                pygame.draw.rect(
                    screen,
                    DARK_GREEN,
                    success_box,
                    border_radius=20
                )


                draw_text(
                    screen,
                    "CORRECTO!",
                    font_title,
                    WHITE,
                    410,
                    292,
                    True
                )


                draw_text(
                    screen,
                    "+100 PUNTOS",
                    font_medium,
                    WHITE,
                    410,
                    332,
                    True
                )


            else:

                success_message_start = None


        # ====================================================
        # GAME OVER
        # ====================================================

        if game_state == "GAME_OVER":

            overlay = pygame.Surface(
                (
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT
                ),
                pygame.SRCALPHA
            )


            overlay.fill(
                (
                    0,
                    0,
                    0,
                    220
                )
            )


            screen.blit(
                overlay,
                (
                    0,
                    0
                )
            )


            draw_text(
                screen,
                "JUEGO TERMINADO",
                font_logo,
                WHITE,
                WINDOW_WIDTH // 2,
                220,
                True
            )


            draw_text(
                screen,
                f"{score} PUNTOS",
                font_logo,
                GREEN,
                WINDOW_WIDTH // 2,
                295,
                True
            )


            draw_text(
                screen,
                f"Posturas correctas: {correct_poses}/{TOTAL_ROUNDS}",
                font_large,
                WHITE,
                WINDOW_WIDTH // 2,
                355,
                True
            )


            percentage = int(
                (
                    correct_poses
                    /
                    TOTAL_ROUNDS
                )
                *
                100
            )


            draw_text(
                screen,
                f"Resultado: {percentage}%",
                font_medium,
                GRAY,
                WINDOW_WIDTH // 2,
                405,
                True
            )


            if percentage >= 80:

                result_text = (
                    "EXCELENTE"
                )

                result_color = (
                    GREEN
                )


            elif percentage >= 50:

                result_text = (
                    "BUEN TRABAJO"
                )

                result_color = (
                    YELLOW
                )


            else:

                result_text = (
                    "SIGUE PRACTICANDO"
                )

                result_color = (
                    RED
                )


            draw_text(
                screen,
                result_text,
                font_large,
                result_color,
                WINDOW_WIDTH // 2,
                465,
                True
            )


            draw_text(
                screen,
                "R = Jugar otra vez",
                font_medium,
                WHITE,
                WINDOW_WIDTH // 2,
                540,
                True
            )


            draw_text(
                screen,
                "Q o ESC = Salir",
                font_small,
                GRAY,
                WINDOW_WIDTH // 2,
                585,
                True
            )


        # ====================================================
        # ACTUALIZAR PANTALLA
        # ====================================================

        pygame.display.flip()


        clock.tick(
            30
        )


# ============================================================
# CERRAR
# ============================================================

cap.release()

pygame.quit()

cv2.destroyAllWindows()