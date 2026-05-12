"""Script de retraining para VinOps.

Mezcla datos originales (wine_clean.csv) + feedback acumulado,
entrena un nuevo RandomForest y guarda el modelo.

Ejecutar: python -m src.train
"""

import os
import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(os.getenv("DATA_PATH", "/app/data/wine_clean.csv"))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "/app/feedback/feedback.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/wine_model.pkl"))

# 13 features en camelCase (como los espera la API)
FEATURE_COLS = [
    "Alcohol",
    "MalicAcid",
    "Ash",
    "AlcalinityAsh",
    "Magnesium",
    "TotalPhenols",
    "Flavanoids",
    "NonflavanoidPhenols",
    "Proanthocyanins",
    "ColorIntensity",
    "Hue",
    "OD280_OD315",
    "Proline",
]
TARGET_COL = "Class"

# Mapeo de snake_case (CSV) a camelCase
CSV_TO_PYDANTIC = {
    "Alcohol": "Alcohol",
    "Malic_Acid": "MalicAcid",
    "Ash": "Ash",
    "Alcalinity_of_Ash": "AlcalinityAsh",
    "Magnesium": "Magnesium",
    "Total_Phenols": "TotalPhenols",
    "Flavanoids": "Flavanoids",
    "Nonflavanoid_Phenols": "NonflavanoidPhenols",
    "Proanthocyanins": "Proanthocyanins",
    "Color_Intensity": "ColorIntensity",
    "Hue": "Hue",
    "OD280_OD315": "OD280_OD315",
    "Proline": "Proline",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas snake_case a camelCase."""
    rename_map = {}
    for col in df.columns:
        if col in CSV_TO_PYDANTIC:
            rename_map[col] = CSV_TO_PYDANTIC[col]
        elif col in FEATURE_COLS or col == TARGET_COL:
            rename_map[col] = col
    return df.rename(columns=rename_map)


def load_original_data() -> pd.DataFrame:
    """Carga el dataset original."""
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": TARGET_COL})
    df = _normalize_columns(df)
    # FILTRADO ESTRICTO: solo 13 features + target
    return df[FEATURE_COLS + [TARGET_COL]]


def load_feedback_data() -> pd.DataFrame:
    """Carga feedback y filtra SOLO features + target."""
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(FEEDBACK_PATH)
    if df.empty:
        return pd.DataFrame()

    # Renombrar true_class -> Class
    if "true_class" in df.columns:
        df = df.rename(columns={"true_class": TARGET_COL})

    df = _normalize_columns(df)

    # FILTRADO ESTRICTO: descartar predicted_class, timestamp, etc.
    available = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Feedback CSV no tiene columnas requeridas: {missing}")

    return df[available]


def train_model() -> Dict[str, object]:
    """Entrena modelo mezclando originales + feedback."""
    print("[RETRAIN] Cargando datos originales...")
    original = load_original_data()
    print(f"[RETRAIN] Originales: {len(original)} muestras, columnas: {list(original.columns)}")

    print("[RETRAIN] Cargando feedback...")
    feedback = load_feedback_data()
    print(f"[RETRAIN] Feedback: {len(feedback)} muestras, columnas: {list(feedback.columns)}")

    if len(feedback) == 0:
        print("[RETRAIN] ERROR: No hay feedback. Abortando.")
        sys.exit(1)

    # Duplicar feedback para dar peso sin dominar
    feedback_weighted = pd.concat([feedback, feedback], ignore_index=True)
    combined = pd.concat([original, feedback_weighted], ignore_index=True)

    # CRITICO: asegurar exactamente 14 columnas (13 + Class)
    combined = combined[FEATURE_COLS + [TARGET_COL]]
    print(f"[RETRAIN] Dataset final: {len(combined)} muestras, {len(combined.columns)} columnas")
    print(f"[RETRAIN] Columnas: {list(combined.columns)}")

    X = combined[FEATURE_COLS]
    y = combined[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_features=2,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    print("[RETRAIN] Entrenando...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"[RETRAIN] F1-Score (macro): {f1:.4f}")
    print(f"[RETRAIN] Balanced Accuracy: {bal_acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"[RETRAIN] Modelo guardado en {MODEL_PATH}")

    return {
        "status": "success",
        "original_size": int(len(original)),
        "feedback_size": int(len(feedback)),
        "combined_size": int(len(combined)),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "model_path": str(MODEL_PATH),
    }


def main() -> None:
    result = train_model()
    print(result)


if __name__ == "__main__":
    main()