"""Almacenamiento de feedback del enologo para retraining.

Formato: CSV con columnas del dataset (snake_case como wine_clean.csv)
+ predicted_class + true_class + timestamp.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "/app/feedback/feedback.csv"))

# Columnas del CSV de feedback
FEEDBACK_COLS = [
    "Alcohol",
    "Malic_Acid",
    "Ash",
    "Alcalinity_of_Ash",
    "Magnesium",
    "Total_Phenols",
    "Flavanoids",
    "Nonflavanoid_Phenols",
    "Proanthocyanins",
    "Color_Intensity",
    "Hue",
    "OD280_OD315",
    "Proline",
    "predicted_class",
    "true_class",
    "timestamp",
]


class FeedbackStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or FEEDBACK_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Crea archivo limpio si no existe o esta corrupto."""
        if not self.path.exists():
            self._write_header()
            return
        
        # Verificar si el CSV actual tiene las columnas correctas
        try:
            df = pd.read_csv(self.path, nrows=0)
            if list(df.columns) != FEEDBACK_COLS:
                print(f"[FEEDBACK] CSV corrupto (columnas: {list(df.columns)}). Recreando...")
                self._write_header()
        except Exception:
            self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLS)
            writer.writeheader()

    def add(self, sample: Dict[str, float], predicted: int, true_class: int) -> Dict[str, Any]:
        """Anade una correccion del enologo al CSV de feedback."""
        row = {
            "Alcohol": sample.get("Alcohol", ""),
            "Malic_Acid": sample.get("Malic_Acid", ""),
            "Ash": sample.get("Ash", ""),
            "Alcalinity_of_Ash": sample.get("Alcalinity_of_Ash", ""),
            "Magnesium": sample.get("Magnesium", ""),
            "Total_Phenols": sample.get("Total_Phenols", ""),
            "Flavanoids": sample.get("Flavanoids", ""),
            "Nonflavanoid_Phenols": sample.get("Nonflavanoid_Phenols", ""),
            "Proanthocyanins": sample.get("Proanthocyanins", ""),
            "Color_Intensity": sample.get("Color_Intensity", ""),
            "Hue": sample.get("Hue", ""),
            "OD280_OD315": sample.get("OD280_OD315", ""),
            "Proline": sample.get("Proline", ""),
            "predicted_class": predicted,
            "true_class": true_class,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLS)
            writer.writerow(row)

        return {"status": "stored", "total_feedback": self.count()}

    def count(self) -> int:
        """Numero de feedbacks acumulados."""
        if not self.path.exists():
            return 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            return max(total_lines - 1, 0)
        except Exception:
            return 0

    def read_all(self) -> pd.DataFrame:
        """Devuelve todos los feedbacks como DataFrame."""
        if not self.path.exists() or self.count() == 0:
            return pd.DataFrame(columns=FEEDBACK_COLS)
        try:
            df = pd.read_csv(self.path)
            # Filtrar solo columnas validas
            return df[[c for c in FEEDBACK_COLS if c in df.columns]]
        except Exception:
            return pd.DataFrame(columns=FEEDBACK_COLS)

    def get_summary(self) -> Dict[str, Any]:
        """Resumen del estado del feedback."""
        df = self.read_all()
        n = len(df)
        if n == 0:
            return {"count": 0, "retrain_recommended": False, "last_feedback": None}

        # Calcular errores solo si las columnas existen
        errors = 0
        if "predicted_class" in df.columns and "true_class" in df.columns:
            errors = int((df["predicted_class"] != df["true_class"]).sum())

        return {
            "count": n,
            "retrain_recommended": n >= 10,
            "retrain_threshold": 10,
            "errors_detected": errors,
            "error_rate": round(errors / n, 3) if n > 0 else 0.0,
            "last_feedback": df["timestamp"].iloc[-1] if n > 0 and "timestamp" in df.columns else None,
            "class_distribution": df["true_class"].value_counts().to_dict() if n > 0 and "true_class" in df.columns else {},
        }

    def clear(self) -> None:
        """Limpia el archivo de feedback."""
        self._write_header()