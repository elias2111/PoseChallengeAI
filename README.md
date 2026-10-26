# Pose Challenge AI

**Pose Challenge AI** es un proyecto de visión artificial y Machine Learning desarrollado en Python que permite reconocer posturas corporales en tiempo real utilizando la cámara de un iPhone.

El sistema utiliza **MediaPipe** para detectar puntos de referencia del cuerpo, un modelo **Random Forest** entrenado con un dataset público de posturas y **Pygame** para convertir el reconocimiento en un juego interactivo.

---

## Objetivo del proyecto

Desarrollar una aplicación capaz de:

- Capturar video en tiempo real utilizando un iPhone.
- Detectar puntos de referencia del cuerpo humano.
- Procesar y normalizar los landmarks corporales.
- Entrenar un modelo propio de clasificación.
- Reconocer posturas corporales en tiempo real.
- Mostrar el nivel de confianza de la predicción.
- Crear un juego interactivo utilizando Pygame.
- Generar un ejecutable portable para Windows.

---

## Tecnologías utilizadas

- Python 3.13
- OpenCV
- MediaPipe
- Scikit-learn
- Random Forest
- NumPy
- Pandas
- Joblib
- Pygame
- Matplotlib
- PyInstaller
- KaggleHub
- Camo Studio
- iPhone

---

## Arquitectura del sistema

```text
iPhone
   ↓
Camo Camera
   ↓
Camo Studio
   ↓
OpenCV
   ↓
MediaPipe Pose Landmarker
   ↓
33 landmarks corporales
   ↓
Normalización
   ↓
99 características
   ↓
Random Forest
   ↓
Clasificación de postura
   ↓
Pygame
   ↓
Pose Challenge AI
```

---

# Dataset

Para entrenar el modelo se utilizó un dataset público de posturas de yoga.

El dataset contiene cinco clases:

```text
downdog
goddess
plank
tree
warrior2
```

Inicialmente se encontraron:

```text
1551 imágenes
```

Después de procesarlas con MediaPipe se obtuvieron:

| Postura | Muestras |
|---|---:|
| Down Dog | 294 |
| Goddess | 252 |
| Plank | 378 |
| Tree | 228 |
| Warrior 2 | 356 |
| **Total** | **1508** |

Se descartaron 43 imágenes en las que MediaPipe no pudo obtener correctamente los landmarks corporales.

---

# Landmarks corporales

MediaPipe Pose Landmarker permite detectar **33 puntos de referencia del cuerpo humano**.

Entre ellos se encuentran:

- Cabeza
- Ojos
- Hombros
- Codos
- Muñecas
- Manos
- Cadera
- Rodillas
- Tobillos
- Pies

Para cada landmark se utilizan tres coordenadas:

```text
X
Y
Z
```

Por lo tanto, cada postura se representa mediante:

```text
33 landmarks × 3 coordenadas = 99 características
```

Estas 99 características son utilizadas como entrada para el modelo de Machine Learning.

---

# Normalización de los datos

Antes de entrenar el modelo, los landmarks son normalizados.

Se utiliza:

```text
Centro del cuerpo = punto medio entre las caderas
```

y:

```text
Escala = longitud del torso
```

Esto permite reducir las diferencias producidas por:

- Distancia entre la persona y la cámara.
- Tamaño de la persona.
- Posición de la persona dentro de la imagen.
- Diferentes resoluciones de cámara.

El objetivo es que el modelo aprenda principalmente la **forma de la postura corporal**.

---

# Modelo de Machine Learning

El clasificador utilizado es:

```python
RandomForestClassifier
```

El modelo recibe como entrada las 99 características correspondientes a los landmarks normalizados.

Las posibles salidas son:

```text
downdog
goddess
plank
tree
warrior2
```

El modelo entrenado se guarda en:

```text
models/pose_classifier.pkl
```

---

# Resultados del entrenamiento

El modelo obtuvo una precisión general de:

```text
Accuracy: 98.49%
```

Resultados obtenidos durante la evaluación:

| Postura | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Down Dog | 100.00% | 98.94% | 99.47% |
| Goddess | 98.70% | 95.00% | 96.82% |
| Plank | 97.46% | 100.00% | 98.71% |
| Tree | 100.00% | 100.00% | 100.00% |
| Warrior 2 | 97.22% | 98.13% | 97.67% |

Cantidad de muestras utilizadas en la prueba:

```text
465
```

La matriz de confusión generada durante la evaluación se encuentra en:

```text
models/confusion_matrix.png
```

---

# Reconocimiento en tiempo real

Después del entrenamiento, el sistema puede utilizar una cámara en tiempo real.

El flujo es:

```text
Persona
   ↓
Cámara del iPhone
   ↓
OpenCV
   ↓
MediaPipe
   ↓
33 landmarks
   ↓
Normalización
   ↓
Random Forest
   ↓
Predicción
```

Ejemplo:

```text
POSTURA: WARRIOR 2
CONFIANZA: 87.4%
```

---

# Uso del iPhone como cámara

El proyecto utiliza un iPhone como dispositivo de captura.

Para conectarlo a Windows se utiliza:

```text
Camo Camera
+
Camo Studio
```

Camo crea una cámara virtual en Windows que puede ser utilizada por OpenCV.

La conexión puede realizarse mediante:

```text
USB
```

o mediante conexión inalámbrica compatible con Camo.

---

# Pose Challenge AI - Juego

El proyecto incluye un juego desarrollado utilizando **Pygame**.

El jugador debe observar la postura que aparece en pantalla e intentar imitarla antes de que termine el tiempo.

El sistema analiza el cuerpo en tiempo real y utiliza el modelo entrenado para determinar si la postura realizada es correcta.

---

## Posturas del juego

Las cinco posturas disponibles son:

### Tree

Postura de equilibrio en una pierna.

```text
TREE
```

### Warrior 2

Brazos extendidos horizontalmente y piernas separadas.

```text
WARRIOR 2
```

### Goddess

Piernas abiertas con rodillas flexionadas.

```text
GODDESS
```

### Plank

Cuerpo recto en posición horizontal.

```text
PLANK
```

### Down Dog

El cuerpo forma aproximadamente una V invertida.

```text
DOWN DOG
```

---

# Reglas del juego

El juego cuenta con:

```text
5 posturas diferentes
10 rondas
15 segundos por postura
100 puntos por acierto
```

Cada postura aparece exactamente:

```text
2 veces
```

durante una partida de 10 rondas.

Para que una postura sea considerada correcta debe mantenerse durante aproximadamente:

```text
0.5 segundos
```

---

# Funciones de la interfaz

La versión final incluye:

- Cámara en tiempo real.
- Esqueleto corporal sobre la persona.
- Postura objetivo.
- Figura de referencia.
- Predicción del modelo.
- Porcentaje de confianza.
- Temporizador.
- Barra de tiempo.
- Barra de mantenimiento de postura.
- Sistema de puntuación.
- Contador de rondas.
- Mensaje de acierto.
- Pantalla inicial.
- Pantalla final de resultados.

---

# Controles

```text
ENTER = iniciar el juego

R = reiniciar después de terminar

Q = salir

ESC = salir
```

---

# Estructura del proyecto

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
├── README.md
└── requirements.txt
```

---

# Archivos principales

## `download_dataset.py`

Descarga el dataset utilizado para entrenar el modelo.

---

## `extract_landmarks.py`

Procesa las imágenes del dataset con MediaPipe y genera:

```text
dataset/landmarks.csv
```

---

## `train_model.py`

Entrena el modelo Random Forest y genera:

```text
models/pose_classifier.pkl
```

y:

```text
models/confusion_matrix.png
```

---

## `pose_detection.py`

Permite visualizar los 33 landmarks corporales detectados por MediaPipe.

---

## `live_prediction.py`

Realiza clasificación de posturas en tiempo real.

---

## `pose_game.py`

Contiene la versión final del juego Pose Challenge AI desarrollado con Pygame.

---

# Instalación desde el código fuente

## 1. Clonar el repositorio

```bash
git clone https://github.com/elias2111/PoseChallengeAI.git
```

Entrar al proyecto:

```bash
cd PoseChallengeAI
```

---

## 2. Crear entorno virtual

```bash
python -m venv .venv
```

---

## 3. Activar entorno virtual en Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecutar detección corporal

```bash
python src/pose_detection.py
```

---

# Ejecutar clasificación en tiempo real

```bash
python src/live_prediction.py
```

---

# Ejecutar el juego

Primero se debe abrir Camo Studio y comprobar que la cámara del iPhone funciona.

Después ejecutar:

```bash
python src/pose_game.py
```

Presionar:

```text
ENTER
```

para comenzar.

---

# Ejecutable para Windows

También existe una versión portable para Windows creada con **PyInstaller**.

No es necesario instalar Python para utilizar esta versión.

## Descargar

La versión ejecutable se encuentra disponible en Google Drive:

[Descargar Pose Challenge AI para Windows](https://drive.google.com/file/d/1Jeg-10gx8nCQLfy_g-fE-l3dYoAmK3Rh/view?usp=drive_link)

Archivo:

```text
PoseChallengeAI_Windows.zip
```

---

# Cómo utilizar el ejecutable

## 1. Descargar

Descargar:

```text
PoseChallengeAI_Windows.zip
```

desde Google Drive.

---

## 2. Extraer

Extraer completamente el archivo ZIP.

No ejecutar el programa directamente dentro del ZIP.

---

## 3. Preparar la cámara

Instalar y abrir:

```text
Camo Studio
```

En el iPhone instalar:

```text
Camo Camera
```

Conectar o emparejar el iPhone con el computador.

Comprobar que Camo Studio muestre correctamente la imagen.

---

## 4. Ejecutar

Entrar a la carpeta:

```text
PoseChallengeAI
```

y ejecutar:

```text
PoseChallengeAI.exe
```

El programa necesita la carpeta:

```text
_internal
```

por lo tanto, **no se debe mover únicamente el archivo `.exe` fuera de la carpeta**.

---

# Requisitos del ejecutable

- Windows 10 o Windows 11.
- Camo Studio.
- iPhone con Camo Camera.
- Cámara correctamente conectada.
- Espacio suficiente para extraer el archivo ZIP.

Python no es necesario para utilizar el ejecutable.

---

# Flujo completo del proyecto

```text
               DATASET DE POSTURAS
                       ↓
                    IMÁGENES
                       ↓
                   MediaPipe
                       ↓
              33 LANDMARKS CORPORALES
                       ↓
                  NORMALIZACIÓN
                       ↓
               99 CARACTERÍSTICAS
                       ↓
                 RANDOM FOREST
                       ↓
                MODELO ENTRENADO
                       ↓
             pose_classifier.pkl


iPhone
   ↓
Camo Studio
   ↓
OpenCV
   ↓
MediaPipe
   ↓
33 landmarks
   ↓
Normalización
   ↓
Random Forest
   ↓
Predicción
   ↓
Pygame
   ↓
POSE CHALLENGE AI
```

---

# Resultado final

El proyecto permite utilizar un dispositivo móvil como cámara y aplicar técnicas de visión artificial y Machine Learning para reconocer posturas humanas en tiempo real.

El sistema combina:

```text
Visión artificial
+
Machine Learning
+
Landmarks corporales
+
Dispositivo móvil
+
Interacción en tiempo real
+
Videojuego
```

El resultado final es una aplicación interactiva capaz de reconocer cinco posturas diferentes y utilizar las predicciones del modelo como mecánica principal de un juego.

---

# Autor

**José Elías**

Ingeniería de Sistemas

---

# Repositorio

GitHub:

https://github.com/elias2111/PoseChallengeAI

---

# Estado del proyecto

```text
✓ iPhone conectado como cámara
✓ OpenCV funcionando
✓ MediaPipe funcionando
✓ 33 landmarks corporales
✓ Dataset público procesado
✓ 1508 muestras válidas
✓ Modelo Random Forest entrenado
✓ Accuracy de 98.49%
✓ Clasificación en tiempo real
✓ Juego desarrollado con Pygame
✓ Sistema de puntuación
✓ Temporizador
✓ Referencias visuales
✓ Ejecutable para Windows
✓ Código publicado en GitHub
✓ Ejecutable disponible en Google Drive
```