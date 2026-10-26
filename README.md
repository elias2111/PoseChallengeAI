# Pose Challenge AI

Pose Challenge AI es un proyecto de visión artificial y aprendizaje automático que reconoce posturas corporales en tiempo real utilizando la cámara de un iPhone.

El sistema utiliza MediaPipe para detectar puntos de referencia del cuerpo, un modelo Random Forest entrenado con un dataset público de posturas y Pygame para convertir el reconocimiento en un juego interactivo.

## Objetivo

Desarrollar una aplicación capaz de:

- Capturar video en tiempo real desde un iPhone.
- Detectar puntos de referencia del cuerpo humano.
- Clasificar diferentes posturas corporales.
- Mostrar la predicción y nivel de confianza.
- Crear un juego de posturas utilizando Pygame.
- Ejecutarse como aplicación portable en Windows.

## Tecnologías utilizadas

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Random Forest
- NumPy
- Pandas
- Pygame
- Joblib
- PyInstaller
- Camo Studio

## Flujo del sistema

```text
iPhone
   ↓
Camo Studio
   ↓
OpenCV
   ↓
MediaPipe
   ↓
33 landmarks corporales
   ↓
Normalización
   ↓
Random Forest
   ↓
Clasificación de postura
   ↓
Pygame
```

## Dataset

Se utilizó el dataset público Yoga Poses Dataset.

El dataset contiene las siguientes cinco clases:

- `downdog`
- `goddess`
- `plank`
- `tree`
- `warrior2`

Se encontraron inicialmente 1551 imágenes.

Después del procesamiento con MediaPipe se obtuvieron:

```text
downdog   -> 294
goddess   -> 252
plank     -> 378
tree      -> 228
warrior2  -> 356
```

Total de muestras válidas:

```text
1508
```

Cada imagen fue procesada utilizando MediaPipe Pose.

MediaPipe devuelve 33 puntos de referencia corporales.

Para cada punto se utilizan las coordenadas:

```text
X
Y
Z
```

Por lo tanto, cada postura es representada mediante:

```text
33 landmarks x 3 coordenadas = 99 características
```

## Normalización

Antes del entrenamiento, los landmarks fueron normalizados.

Se utilizó el centro de las caderas como punto de referencia y la longitud del torso como escala.

Esto permite reducir el efecto de:

- Distancia a la cámara.
- Tamaño de la persona.
- Posición dentro de la imagen.

## Modelo de Machine Learning

Se entrenó un modelo:

```text
RandomForestClassifier
```

El modelo utiliza los 99 valores obtenidos de los landmarks como entrada y predice una de las cinco posturas.

El modelo entrenado se encuentra en:

```text
models/pose_classifier.pkl
```

## Resultados

El modelo obtuvo una precisión general de:

```text
Accuracy: 98.49%
```

Resultados aproximados por clase:

| Postura | F1-score |
|---|---:|
| Down Dog | 99.47% |
| Goddess | 96.82% |
| Plank | 98.71% |
| Tree | 100% |
| Warrior 2 | 97.67% |

La matriz de confusión se encuentra en:

```text
models/confusion_matrix.png
```

## Juego

El proyecto incluye un juego creado con Pygame.

El jugador debe realizar la postura mostrada en pantalla antes de que termine el tiempo.

Características:

- 10 rondas.
- 5 posturas.
- Cada postura aparece exactamente dos veces.
- 15 segundos por postura.
- 100 puntos por acierto.
- Reconocimiento en tiempo real.
- Indicador de confianza.
- Referencia visual de cada postura.
- Barra de tiempo.
- Pantalla de resultados.

Las posturas utilizadas son:

```text
TREE
WARRIOR 2
GODDESS
PLANK
DOWN DOG
```

## Cámara

El proyecto fue desarrollado utilizando un iPhone como cámara.

Para conectar el iPhone con Windows se utiliza:

```text
Camo Camera
+
Camo Studio
```

OpenCV reconoce posteriormente la cámara virtual generada por Camo.

## Estructura del proyecto

```text
PoseChallengeAI/
│
├── dataset/
│   └── landmarks.csv
│
├── models/
│   ├── confusion_matrix.png
│   ├── pose_classifier.pkl
│   └── pose_landmarker_lite.task
│
├── src/
│   ├── download_dataset.py
│   ├── extract_landmarks.py
│   ├── live_prediction.py
│   ├── pose_detection.py
│   ├── pose_game.py
│   ├── test_camera.py
│   └── train_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar detección de cuerpo

```bash
python src/pose_detection.py
```

## Ejecutar clasificación en tiempo real

```bash
python src/live_prediction.py
```

## Ejecutar el juego

```bash
python src/pose_game.py
```

## Controles del juego

```text
ENTER = iniciar juego
R = reiniciar al terminar
Q = salir
ESC = salir
```

## Ejecutable para Windows

El proyecto también fue empaquetado utilizando PyInstaller.

La versión ejecutable necesita Camo Studio para utilizar el iPhone como cámara.

La carpeta completa del ejecutable puede almacenarse en Google Drive para descargarla y ejecutarla en otro computador.

## Autor

**José Elías**

Ingeniería de Sistemas

## Estado del proyecto

```text
✓ Dataset procesado
✓ MediaPipe funcionando
✓ Modelo entrenado
✓ Accuracy 98.49%
✓ Clasificación en tiempo real
✓ Juego en Pygame
✓ Cámara mediante iPhone
✓ Ejecutable de Windows
```