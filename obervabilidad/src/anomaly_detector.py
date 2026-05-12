"""
Detector de anomalías basado en umbrales estadísticos del dataset de entrenamiento.
Reutiliza la lógica de Hito 2 (±3σ por variable) adaptada a inferencia online.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

class AnomalyDetector:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.getenv("DATA_PATH", "/app/data/wine_clean.csv")
        self.stats = self._build_reference_stats()

    def _build_reference_stats(self):
        """Calcula medias y desviaciones globales del dataset de referencia."""
        df = pd.read_csv(self.data_path)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        # Solo variables numéricas de entrada
        stats = {
            "means": df.mean().to_dict(),
            "stds": df.std().to_dict(),
            "columns": list(df.columns)
        }
        return stats

    def check(self, sample: dict) -> dict:
        """
        Recibe un dict con las 13 variables.
        Devuelve: {
            'is_anomaly': bool,
            'anomalous_variables': list[str],
            'z_scores': dict[str, float]
        }
        """
        anomalous = []
        z_scores = {}

        for col in self.stats["columns"]:
            if col not in sample:
                continue
            mean = self.stats["means"][col]
            std = self.stats["stds"][col]
            if std == 0:
                z = 0.0
            else:
                z = (sample[col] - mean) / std
            z_scores[col] = round(z, 3)
            if abs(z) > 3.0:
                anomalous.append(col)

        return {
            "is_anomaly": len(anomalous) > 0,
            "anomalous_variables": anomalous,
            "z_scores": z_scores,
            "threshold": 3.0
        }
