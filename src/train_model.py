import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# CONFIGURACION
# ==========================================

DATASET_FILE = Path("dataset/landmarks.csv")

MODEL_FILE = Path(
    "models/pose_classifier.pkl"
)

MATRIX_FILE = Path(
    "models/confusion_matrix.png"
)


# ==========================================
# CARGAR DATASET
# ==========================================

print("======================================")
print("POSE CHALLENGE AI")
print("ENTRENAMIENTO DEL MODELO")
print("======================================")

df = pd.read_csv(DATASET_FILE)

print(
    f"\nMuestras disponibles: {len(df)}"
)

print("\nClases:")

print(
    df["label"].value_counts()
)


# ==========================================
# CARACTERISTICAS
# ==========================================

feature_columns = []

for i in range(33):

    feature_columns.append(f"x{i}")
    feature_columns.append(f"y{i}")
    feature_columns.append(f"z{i}")


X = df[feature_columns]

y = df["label"]


# ==========================================
# TRAIN / TEST
# ==========================================

# Si el dataset original ya trae
# train y test, los respetamos.

if (
    "split" in df.columns
    and
    "train" in df["split"].values
    and
    "test" in df["split"].values
):

    print(
        "\nUsando division TRAIN / TEST "
        "original del dataset."
    )

    train_df = df[
        df["split"] == "train"
    ]

    test_df = df[
        df["split"] == "test"
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "label"
    ]

else:

    print(
        "\nNo se encontro division original."
    )

    print(
        "Usando 80% entrenamiento "
        "y 20% prueba."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


print(
    f"\nEntrenamiento: {len(X_train)} muestras"
)

print(
    f"Prueba: {len(X_test)} muestras"
)


# ==========================================
# MODELO
# ==========================================

print(
    "\nEntrenando Random Forest..."
)

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Entrenamiento terminado."
)


# ==========================================
# EVALUACION
# ==========================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n======================================")
print("RESULTADOS")
print("======================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


# ==========================================
# MATRIZ DE CONFUSION
# ==========================================

labels = sorted(
    y.unique()
)

matrix = confusion_matrix(

    y_test,

    predictions,

    labels=labels
)


display = ConfusionMatrixDisplay(

    confusion_matrix=matrix,

    display_labels=labels
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Pose Challenge AI - Matriz de Confusion"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    MATRIX_FILE,
    dpi=200
)

plt.close()


# ==========================================
# GUARDAR MODELO
# ==========================================

model_data = {

    "model": model,

    "features": feature_columns,

    "labels": labels
}


joblib.dump(

    model_data,

    MODEL_FILE
)


print("\n======================================")

print(
    "MODELO GUARDADO CORRECTAMENTE"
)

print(
    "======================================"
)

print(
    f"\nModelo: {MODEL_FILE}"
)

print(
    f"Matriz: {MATRIX_FILE}"
)