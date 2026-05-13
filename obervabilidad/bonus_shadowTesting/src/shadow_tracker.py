"""Tracker de comparaciones Champion vs Challenger (Shadow Testing).

Acumula predicciones de ambos modelos sobre las mismas muestras,
permitiendo evaluar al Challenger sin afectar al cliente.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SHADOW_LOG_PATH = Path(os.getenv("SHADOW_LOG_PATH", "/app/shadow_logs/comparisons.csv"))

SHADOW_COLS = [
    "timestamp",
    "champion_prediction",
    "challenger_prediction",
    "champion_confidence",
    "challenger_confidence",
    "agreement",  # True si coinciden, False si no
    "confidence_delta",  # challenger - champion
    "latency_champion_ms",
    "latency_challenger_ms",
    "features_hash",  # hash simple de las features para trazabilidad
]


class ShadowTracker:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or SHADOW_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHADOW_COLS)
            writer.writeheader()

    def log_comparison(
        self,
        champion_pred: int,
        challenger_pred: int,
        champion_conf: float,
        challenger_conf: float,
        latency_champion_ms: float,
        latency_challenger_ms: float,
        features_hash: str,
    ) -> Dict[str, Any]:
        """Registra una comparacion Champion vs Challenger."""
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "champion_prediction": champion_pred,
            "challenger_prediction": challenger_pred,
            "champion_confidence": round(champion_conf, 4),
            "challenger_confidence": round(challenger_conf, 4),
            "agreement": champion_pred == challenger_pred,
            "confidence_delta": round(challenger_conf - champion_conf, 4),
            "latency_champion_ms": round(latency_champion_ms, 2),
            "latency_challenger_ms": round(latency_challenger_ms, 2),
            "features_hash": features_hash,
        }

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHADOW_COLS)
            writer.writerow(row)

        return {"status": "logged", "agreement": row["agreement"]}

    def get_summary(self) -> Dict[str, Any]:
        """Resumen estadistico de las comparaciones acumuladas."""
        if not self.path.exists():
            return {"comparisons": 0, "champion_loaded": False, "challenger_loaded": False}

        df = pd.read_csv(self.path)
        n = len(df)
        if n == 0:
            return {"comparisons": 0, "champion_loaded": True, "challenger_loaded": True}

        agreements = df["agreement"].sum()
        avg_conf_champion = df["champion_confidence"].mean()
        avg_conf_challenger = df["challenger_confidence"].mean()
        avg_latency_champion = df["latency_champion_ms"].mean()
        avg_latency_challenger = df["latency_challenger_ms"].mean()

        return {
            "comparisons": n,
            "agreements": int(agreements),
            "disagreements": int(n - agreements),
            "agreement_rate": round(agreements / n, 3) if n > 0 else 0.0,
            "avg_confidence_champion": round(float(avg_conf_champion), 4),
            "avg_confidence_challenger": round(float(avg_conf_challenger), 4),
            "avg_confidence_delta": round(float(avg_conf_challenger - avg_conf_champion), 4),
            "avg_latency_champion_ms": round(float(avg_latency_champion), 2),
            "avg_latency_challenger_ms": round(float(avg_latency_challenger), 2),
            "latency_delta_ms": round(float(avg_latency_challenger - avg_latency_champion), 2),
            "champion_better_confidence": int((df["confidence_delta"] < 0).sum()),
            "challenger_better_confidence": int((df["confidence_delta"] > 0).sum()),
        }

    def clear(self) -> None:
        """Limpia el log de comparaciones."""
        self._write_header()