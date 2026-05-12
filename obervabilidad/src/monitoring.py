"""
Módulo de observabilidad para VinOps.
Métricas operativas (Prometheus) + métricas proxy del modelo.
"""
import time
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
import json
import os
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Métricas Operativas (Prometheus)
# ------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "vinops_requests_total",
    "Total de peticiones",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "vinops_request_latency_seconds",
    "Latencia de peticiones",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
CPU_USAGE = Gauge(
    "vinops_cpu_usage_percent",
    "Uso de CPU del proceso (%)"
)
MEMORY_USAGE = Gauge(
    "vinops_memory_usage_bytes",
    "Uso de RAM del proceso (bytes)"
)

# ------------------------------------------------------------------
# Métricas Proxy del Modelo (Prometheus)
# ------------------------------------------------------------------
PREDICTION_CONFIDENCE = Histogram(
    "vinops_prediction_confidence",
    "Confianza máxima de la predicción (probabilidad)",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)
PREDICTION_ENTROPY = Histogram(
    "vinops_prediction_entropy",
    "Entropía de la distribución de probabilidades",
    buckets=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
)
PREDICTIONS_BY_CLASS = Counter(
    "vinops_predictions_by_class_total",
    "Predicciones acumuladas por clase",
    ["class_id"]
)
ANOMALY_FLAG = Counter(
    "vinops_anomaly_flags_total",
    "Muestras marcadas como anómalas",
    ["trigger"]
)
LOW_CONFIDENCE_FLAG = Counter(
    "vinops_low_confidence_total",
    "Predicciones con confianza < 0.70"
)

# ------------------------------------------------------------------
# Estado en memoria para /monitor (resumen agregado)
# ------------------------------------------------------------------
class MonitorState:
    def __init__(self):
        self.predictions_count = 0
        self.anomaly_count = 0
        self.low_confidence_count = 0
        self.class_counts = {1: 0, 2: 0, 3: 0}
        self.confidence_sum = 0.0
        self.start_time = datetime.utcnow()

    def register_prediction(self, class_id: int, confidence: float, is_anomaly: bool, is_low_conf: bool):
        self.predictions_count += 1
        self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1
        self.confidence_sum += confidence
        if is_anomaly:
            self.anomaly_count += 1
        if is_low_conf:
            self.low_confidence_count += 1

    def get_summary(self):
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_conf = self.confidence_sum / self.predictions_count if self.predictions_count else 0.0
        return {
            "predictions_total": self.predictions_count,
            "anomalies_total": self.anomaly_count,
            "low_confidence_total": self.low_confidence_count,
            "avg_confidence": round(avg_conf, 4),
            "class_distribution": self.class_counts,
            "uptime_seconds": round(uptime, 1)
        }

MONITOR_STATE = MonitorState()

# ------------------------------------------------------------------
# Logging estructurado JSON
# ------------------------------------------------------------------
LOG_PATH = Path(os.getenv("LOG_PATH", "/app/logs"))
LOG_PATH.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    LOG_PATH / "vinops_predictions.jsonl",
    serialize=True,
    rotation="10 MB",
    retention="7 days",
    enqueue=True
)

def log_prediction(payload: dict):
    """Escribe una línea JSON con toda la trazabilidad de la predicción."""
    logger.info(payload)

def update_system_metrics():
    """Actualiza CPU y RAM del proceso."""
    process = psutil.Process()
    CPU_USAGE.set(process.cpu_percent(interval=0.1))
    MEMORY_USAGE.set(process.memory_info().rss)

def compute_entropy(probas: np.ndarray) -> float:
    """Entropía Shannon (bits) de un vector de probabilidades."""
    probas = np.array(probas)
    probas = probas[probas > 0]
    return float(-np.sum(probas * np.log2(probas)))
