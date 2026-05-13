"""Script de entrenamiento del Challenger para Shadow Testing.

Genera wine_model_challenger.pkl sin tocar el Champion.
Se ejecuta desde el Hito 5 tras acumular feedback, luego se copia aqui.
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
CHALLENGER_PATH = Path(os.getenv("CHALLENGER_PATH", "/app/models/wine_model_challenger.pkl"))

FEATURE_COLS = [
    "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium",
    "TotalPhenols", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins",
    "ColorIntensity", "Hue", "OD280_OD315", "Proline",
]
TARGET_COL = "Class"


def load_data() -> pd.DataFrame:
    """Carga originales + feedback, filtra columnas exactas."""
    df_orig = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df_orig.columns and "class" in df_orig.columns:
        df_orig = df_orig.rename(columns={"class": TARGET_COL})
    
    # Renombrar snake_case a camelCase si viene del CSV original
    rename_map = {
        "Malic_Acid": "MalicAcid",
        "Alcalinity_of_Ash": "AlcalinityAsh",
        "Total_Phenols": "TotalPhenols",
        "Nonflavanoid_Phenols": "NonflavanoidPhenols",
        "Color_Intensity": "ColorIntensity",
    }
    df_orig = df_orig.rename(columns=rename_map)
    
    # Filtrar exactamente las columnas necesarias
    df_orig = df_orig[[c for c in FEATURE_COLS + [TARGET_COL] if c in df_orig.columns]]
    
    # Cargar feedback si existe
    if FEEDBACK_PATH.exists():
        df_fb = pd.read_csv(FEEDBACK_PATH)
        if "true_class" in df_fb.columns:
            df_fb = df_fb.rename(columns={"true_class": TARGET_COL})
        # Renombrar y filtrar
        df_fb = df_fb.rename(columns=rename_map)
        df_fb = df_fb[[c for c in FEATURE_COLS + [TARGET_COL] if c in df_fb.columns]]
        
        # Duplicar feedback x2 para peso
        df_fb = pd.concat([df_fb, df_fb], ignore_index=True)
        combined = pd.concat([df_orig, df_fb], ignore_index=True)
    else:
        combined = df_orig
    
    return combined[FEATURE_COLS + [TARGET_COL]]


def train_challenger() -> Dict[str, object]:
    """Entrena el Challenger y lo guarda separado del Champion."""
    print("[CHALLENGER] Cargando datos...")
    df = load_data()
    print(f"[CHALLENGER] Dataset: {len(df)} muestras, columnas: {list(df.columns)}")
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
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
    
    print("[CHALLENGER] Entrenando...")
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"[CHALLENGER] F1-Score: {f1:.4f}")
    print(f"[CHALLENGER] Balanced Accuracy: {bal_acc:.4f}")
    
    CHALLENGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, CHALLENGER_PATH)
    print(f"[CHALLENGER] Guardado en {CHALLENGER_PATH}")
    
    return {
        "status": "success",
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "model_path": str(CHALLENGER_PATH),
    }


if __name__ == "__main__":
    result = train_challenger()
    print(result)