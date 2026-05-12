"""
VinOps – Servicio de inferencia con observabilidad completa (Fase 1 Hito 5).
Endpoints:
  GET  /health    → Estado del servicio + resumen de monitorización
  POST /predict   → Predicción + métricas proxy + logging estructurado
  GET  /metrics   → Métricas Prometheus (text/plain)
  GET  /monitor   → Resumen agregado de predicciones (JSON)
"""
import os
import time
import pickle
import joblib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.monitoring import (
    REQUEST_COUNT, REQUEST_LATENCY, CPU_USAGE, MEMORY_USAGE,
    PREDICTION_CONFIDENCE, PREDICTION_ENTROPY, PREDICTIONS_BY_CLASS,
    ANOMALY_FLAG, LOW_CONFIDENCE_FLAG,
    update_system_metrics, compute_entropy, log_prediction, MONITOR_STATE
)
from src.anomaly_detector import AnomalyDetector

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/wine_model.pkl"))
DATA_PATH = os.getenv("DATA_PATH", "/app/data/wine_clean.csv")

# ------------------------------------------------------------------
# Carga del modelo
# ------------------------------------------------------------------
app = FastAPI(title="VinOps Inference – Monitored", version="5.1.0")

model = None
model_metadata = {}

@app.on_event("startup")
def load_model():
    global model, model_metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. "
                           f"Copia el artefacto entrenado (wine_model.pkl) a models/")

    # Intentar joblib primero, luego pickle
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    model_metadata = {
        "model_path": str(MODEL_PATH),
        "model_type": type(model).__name__,
        "loaded_at": pd.Timestamp.now().isoformat(),
    }

    # Inicializar detector de anomalías
    app.state.anomaly_detector = AnomalyDetector(data_path=DATA_PATH)

    # Métricas base del sistema
    update_system_metrics()

# ------------------------------------------------------------------
# Esquema de entrada (13 variables fisicoquímicas)
# ------------------------------------------------------------------
class WineFeatures(BaseModel):
    Alcohol: float = Field(..., ge=10.0, le=16.0, description="Contenido alcohólico (% vol)")
    MalicAcid: float = Field(..., ge=0.0, le=6.0, description="Ácido málico (g/L)")
    Ash: float = Field(..., ge=1.0, le=4.0, description="Cenizas (g/L)")
    AlcalinityAsh: float = Field(..., ge=10.0, le=35.0, description="Alcalinidad cenizas (mEq/L)")
    Magnesium: float = Field(..., ge=60.0, le=180.0, description="Magnesio (mg/L)")
    TotalPhenols: float = Field(..., ge=0.5, le=4.0, description="Fenoles totales (g/L)")
    Flavanoids: float = Field(..., ge=0.0, le=6.0, description="Flavanoides (g/L)")
    NonflavanoidPhenols: float = Field(..., ge=0.0, le=1.0, description="Fenoles no flavonoides (g/L)")
    Proanthocyanins: float = Field(..., ge=0.0, le=4.0, description="Proantocianidinas (g/L)")
    ColorIntensity: float = Field(..., ge=1.0, le=14.0, description="Intensidad color (u.a.)")
    Hue: float = Field(..., ge=0.3, le=2.0, description="Tonalidad (u.a.)")
    OD280_OD315: float = Field(..., ge=1.0, le=4.5, description="Ratio OD280/OD315 (u.a.)")
    Proline: float = Field(..., ge=200.0, le=1700.0, description="Prolina (mg/L)")

    class Config:
        json_schema_extra = {
            "example": {
                "Alcohol": 13.2,
                "MalicAcid": 1.78,
                "Ash": 2.14,
                "AlcalinityAsh": 11.2,
                "Magnesium": 100,
                "TotalPhenols": 2.65,
                "Flavanoids": 2.76,
                "NonflavanoidPhenols": 0.26,
                "Proanthocyanins": 1.28,
                "ColorIntensity": 4.38,
                "Hue": 1.05,
                "OD280_OD315": 3.40,
                "Proline": 1050
            }
        }

class PredictionOut(BaseModel):
    prediction: int
    class_label: str
    probabilities: Dict[str, float]
    confidence: float
    entropy: float
    anomaly: Dict[str, Any]
    is_low_confidence: bool
    model_version: str
    latency_ms: float

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    update_system_metrics()
    summary = MONITOR_STATE.get_summary()
    return {
        "status": "ok",
        "service": "vinops-inference-monitored",
        "model": model_metadata,
        "monitor_summary": summary,
        "cpu_percent": CPU_USAGE._value.get(),
        "memory_bytes": MEMORY_USAGE._value.get()
    }

@app.post("/predict", response_model=PredictionOut)
def predict(features: WineFeatures):
    start = time.perf_counter()

    # 1. Preparar muestra
    sample_dict = features.model_dump()
    sample_df = pd.DataFrame([sample_dict])

    # Asegurar orden de columnas que espera el modelo
    expected_cols = [
        "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium",
        "TotalPhenols", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins",
        "ColorIntensity", "Hue", "OD280_OD315", "Proline"
    ]
    sample_df = sample_df[expected_cols]

    # 2. Predicción
    try:
        pred_class = int(model.predict(sample_df)[0])
        probas = model.predict_proba(sample_df)[0]
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {e}")

    confidence = float(np.max(probas))
    entropy = compute_entropy(probas)
    probas_dict = {f"class_{i+1}": round(float(p), 4) for i, p in enumerate(probas)}

    # 3. Detección de anomalías
    anomaly_result = app.state.anomaly_detector.check(sample_dict)

    # 4. Flags de calidad
    is_low_conf = confidence < 0.70

    # 5. Actualizar métricas Prometheus
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

    PREDICTION_CONFIDENCE.observe(confidence)
    PREDICTION_ENTROPY.observe(entropy)
    PREDICTIONS_BY_CLASS.labels(class_id=str(pred_class)).inc()

    if anomaly_result["is_anomaly"]:
        ANOMALY_FLAG.labels(trigger="zscore_3sigma").inc()
    if is_low_conf:
        LOW_CONFIDENCE_FLAG.inc()

    update_system_metrics()

    # 6. Estado agregado
    MONITOR_STATE.register_prediction(
        class_id=pred_class,
        confidence=confidence,
        is_anomaly=anomaly_result["is_anomaly"],
        is_low_conf=is_low_conf
    )

    # 7. Logging estructurado JSON
    log_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "endpoint": "/predict",
        "input": sample_dict,
        "prediction": pred_class,
        "probabilities": probas_dict,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "anomaly": anomaly_result,
        "is_low_confidence": is_low_conf,
        "latency_ms": round(latency * 1000, 2)
    }
    log_prediction(log_payload)

    return PredictionOut(
        prediction=pred_class,
        class_label=f"Cultivar_{pred_class}",
        probabilities=probas_dict,
        confidence=round(confidence, 4),
        entropy=round(entropy, 4),
        anomaly=anomaly_result,
        is_low_confidence=is_low_conf,
        model_version=model_metadata.get("loaded_at", "unknown"),
        latency_ms=round(latency * 1000, 2)
    )

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    update_system_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/monitor")
def monitor():
    return {
        "service": "vinops-observability",
        "model_loaded": model_metadata.get("loaded_at"),
        "current_metrics": MONITOR_STATE.get_summary(),
        "system": {
            "cpu_percent": CPU_USAGE._value.get(),
            "memory_bytes": MEMORY_USAGE._value.get()
        }
    }
