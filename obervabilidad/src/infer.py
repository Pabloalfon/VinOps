"""VinOps – Servicio de inferencia completo (Fases 1+2+3+4+5 Hito 5).

Incluye: observabilidad, drift, alertas, simulacion, feedback loop y retraining.

Este modulo debe exponer `app` (FastAPI) para `uvicorn src.infer:app`.
"""

import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.alerts import AlertManager
from src.anomaly_detector import AnomalyDetector
from src.drift_detector import DriftDetector
from src.feedback_store import FeedbackStore
from src.monitoring import (
    ANOMALY_FLAG,
    CPU_USAGE,
    LOW_CONFIDENCE_FLAG,
    MEMORY_USAGE,
    MONITOR_STATE,
    PREDICTION_CONFIDENCE,
    PREDICTION_ENTROPY,
    PREDICTIONS_BY_CLASS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    compute_entropy,
    log_prediction,
    update_system_metrics,
)


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/wine_model.pkl"))
DATA_PATH = os.getenv("DATA_PATH", "/app/data/wine_clean.csv")
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "/app/feedback/feedback.csv"))

CPU_ALERT_THRESHOLD = float(os.getenv("CPU_ALERT_THRESHOLD", "80.0"))
MEM_ALERT_THRESHOLD_MB = float(os.getenv("MEM_ALERT_THRESHOLD_MB", "300.0"))
CONFIDENCE_ALERT_THRESHOLD = float(os.getenv("CONFIDENCE_ALERT_THRESHOLD", "0.65"))
ANOMALY_RATE_ALERT_THRESHOLD = float(os.getenv("ANOMALY_RATE_ALERT_THRESHOLD", "0.25"))


PYDANTIC_TO_CSV = {
    "Alcohol": "Alcohol",
    "MalicAcid": "Malic_Acid",
    "Ash": "Ash",
    "AlcalinityAsh": "Alcalinity_of_Ash",
    "Magnesium": "Magnesium",
    "TotalPhenols": "Total_Phenols",
    "Flavanoids": "Flavanoids",
    "NonflavanoidPhenols": "Nonflavanoid_Phenols",
    "Proanthocyanins": "Proanthocyanins",
    "ColorIntensity": "Color_Intensity",
    "Hue": "Hue",
    "OD280_OD315": "OD280_OD315",
    "Proline": "Proline",
}
MODEL_COLS = list(PYDANTIC_TO_CSV.keys())


app = FastAPI(title="VinOps Inference – Full v5.5", version="5.5.0")

model = None
model_metadata: Dict[str, Any] = {}
anomaly_detector: AnomalyDetector | None = None
drift_detector: DriftDetector | None = None
alert_manager: AlertManager | None = None
feedback_store: FeedbackStore | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, model_metadata, anomaly_detector, drift_detector, alert_manager, feedback_store

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}.")

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

    anomaly_detector = AnomalyDetector(data_path=DATA_PATH)
    drift_detector = DriftDetector(data_path=DATA_PATH, window_size=20, eval_every=10)
    alert_manager = AlertManager()
    feedback_store = FeedbackStore(path=FEEDBACK_PATH)

    update_system_metrics()


class WineFeatures(BaseModel):
    Alcohol: float = Field(..., ge=10.0, le=16.0)
    MalicAcid: float = Field(..., ge=0.0, le=6.0)
    Ash: float = Field(..., ge=1.0, le=4.0)
    AlcalinityAsh: float = Field(..., ge=10.0, le=35.0)
    Magnesium: float = Field(..., ge=60.0, le=180.0)
    TotalPhenols: float = Field(..., ge=0.5, le=4.0)
    Flavanoids: float = Field(..., ge=0.0, le=6.0)
    NonflavanoidPhenols: float = Field(..., ge=0.0, le=1.0)
    Proanthocyanins: float = Field(..., ge=0.0, le=4.0)
    ColorIntensity: float = Field(..., ge=1.0, le=14.0)
    Hue: float = Field(..., ge=0.3, le=2.0)
    OD280_OD315: float = Field(..., ge=1.0, le=4.5)
    Proline: float = Field(..., ge=200.0, le=1700.0)


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


class FeedbackIn(BaseModel):
    features: WineFeatures
    predicted_class: int = Field(..., ge=1, le=3)
    true_class: int = Field(..., ge=1, le=3)


class SimulateAnomalyIn(BaseModel):
    count: int = Field(5, ge=1, le=20)


class SimulateDriftIn(BaseModel):
    count: int = Field(20, ge=10, le=50)
    shift_variable: str = Field("Alcohol")
    shift_sigma: float = Field(3.0, ge=1.0, le=5.0)


def _clean_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _to_csv_names(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {PYDANTIC_TO_CSV.get(k, k): v for k, v in sample_dict.items()}


def _to_pydantic_names(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    reverse = {v: k for k, v in PYDANTIC_TO_CSV.items()}
    return {reverse.get(k, k): v for k, v in sample_dict.items()}


def _load_model_internal() -> None:
    """Recarga el modelo desde disco."""
    global model, model_metadata
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

# ------------------------------------------------------------------
# Endpoints existentes (Fases 1-4)
# ------------------------------------------------------------------

@app.get("/health")
def health():
    update_system_metrics()
    summary = MONITOR_STATE.get_summary()
    drift_summary = drift_detector.get_summary() if drift_detector else {"status": "NOT_LOADED"}
    fb_summary = feedback_store.get_summary() if feedback_store else {"count": 0}
    return {
        "status": "ok",
        "service": "vinops-inference-monitored",
        "version": "5.5.0",
        "model": model_metadata,
        "monitor_summary": summary,
        "drift_summary": drift_summary,
        "feedback_summary": fb_summary,
        "cpu_percent": float(CPU_USAGE._value.get() or 0.0),
        "memory_bytes": float(MEMORY_USAGE._value.get() or 0.0),
        "alerts_enabled": alert_manager.enabled if alert_manager else False,
    }

@app.post("/predict", response_model=PredictionOut)
def predict(features: WineFeatures):
    start = time.perf_counter()
    sample_dict = features.model_dump()
    
    # Asegurar orden exacto de columnas para el modelo
    sample_df = pd.DataFrame([sample_dict])[MODEL_COLS]
    
    try:
        # Usar .to_numpy() para evitar validacion de nombres de columnas
        pred_class = int(model.predict(sample_df.to_numpy())[0])
        probas = model.predict_proba(sample_df.to_numpy())[0]
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {e}")
    
    confidence = float(np.max(probas))
    entropy = compute_entropy(probas)
    probas_dict = {f"class_{i+1}": round(float(p), 4) for i, p in enumerate(probas)}
    
    csv_sample = _to_csv_names(sample_dict)
    anomaly_result = anomaly_detector.check(csv_sample)
    is_low_conf = confidence < 0.70
    
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
    
    MONITOR_STATE.register_prediction(
        class_id=pred_class, confidence=confidence,
        is_anomaly=anomaly_result["is_anomaly"], is_low_conf=is_low_conf
    )
    
    drift_detector.add_sample(csv_sample)
    
    drift_eval = None
    if drift_detector.should_evaluate():
        drift_eval = drift_detector.evaluate()
        if drift_eval["status"] == "DRIFT_DETECTED":
            alert_manager.send_drift_alert(drift_eval)
        summary = MONITOR_STATE.get_summary()
        avg_conf = summary.get("avg_confidence", 1.0)
        if avg_conf < CONFIDENCE_ALERT_THRESHOLD:
            alert_manager.send_low_confidence_alert(avg_conf, summary.get("predictions_total", 1))
        anomaly_rate = summary.get("anomalies_total", 0) / max(summary.get("predictions_total", 1), 1)
        if anomaly_rate > ANOMALY_RATE_ALERT_THRESHOLD:
            alert_manager.send_anomaly_alert(summary.get("anomalies_total", 0), summary.get("predictions_total", 1))
    
    cpu_val = float(CPU_USAGE._value.get() or 0.0)
    if cpu_val > CPU_ALERT_THRESHOLD:
        alert_manager.send_operational_alert("CPU %", cpu_val, CPU_ALERT_THRESHOLD)
    
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
        "latency_ms": round(latency * 1000, 2),
        "drift_evaluated": drift_eval is not None,
        "drift_status": drift_eval.get("status") if drift_eval else None,
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
    drift_summary = drift_detector.get_summary() if drift_detector else {"status": "NOT_LOADED"}
    drift_summary = _clean_nan(drift_summary)
    fb_summary = feedback_store.get_summary() if feedback_store else {"count": 0}
    return {
        "service": "vinops-observability",
        "version": "5.5.0",
        "model_loaded": model_metadata.get("loaded_at"),
        "current_metrics": MONITOR_STATE.get_summary(),
        "drift": drift_summary,
        "feedback": fb_summary,
        "system": {
            "cpu_percent": float(CPU_USAGE._value.get() or 0.0),
            "memory_bytes": float(MEMORY_USAGE._value.get() or 0.0),
        },
        "alerts": {
            "enabled": alert_manager.enabled if alert_manager else False,
            "throttle_seconds": alert_manager.throttle_seconds if alert_manager else 0,
        }
    }

@app.get("/drift")
def drift_status():
    summary = drift_detector.get_summary() if drift_detector else {"status": "NOT_LOADED"}
    return _clean_nan(summary)

@app.post("/simulate/anomaly")
def simulate_anomaly(payload: SimulateAnomalyIn):
    results = []
    cols = anomaly_detector.stats["columns"]
    np.random.seed(42)
    ranges = {
        "Alcohol": (10.0, 16.0), "Malic_Acid": (0.0, 6.0), "Ash": (1.0, 4.0),
        "Alcalinity_of_Ash": (10.0, 35.0), "Magnesium": (60.0, 180.0),
        "Total_Phenols": (0.5, 4.0), "Flavanoids": (0.0, 6.0),
        "Nonflavanoid_Phenols": (0.0, 1.0), "Proanthocyanins": (0.0, 4.0),
        "Color_Intensity": (1.0, 14.0), "Hue": (0.3, 2.0),
        "OD280_OD315": (1.0, 4.5), "Proline": (200.0, 1700.0),
    }
    for i in range(payload.count):
        csv_sample = {}
        for col in cols:
            mean = anomaly_detector.stats["means"][col]
            std = anomaly_detector.stats["stds"][col]
            direction = 1 if i % 2 == 0 else -1
            val = mean + direction * 2.5 * std
            low, high = ranges.get(col, (0.1, 9999))
            csv_sample[col] = float(np.clip(val, low, high))
        pydantic_sample = _to_pydantic_names(csv_sample)
        df = pd.DataFrame([pydantic_sample])[MODEL_COLS]
        try:
            pred = int(model.predict(df.to_numpy())[0])
            probas = model.predict_proba(df.to_numpy())[0]
            conf = float(np.max(probas))
        except Exception:
            pred = -1; conf = 0.0
        anom = anomaly_detector.check(csv_sample)
        drift_detector.add_sample(csv_sample)
        results.append({"sample_id": i, "prediction": pred, "confidence": round(conf, 3), "is_anomaly": anom["is_anomaly"], "anomalous_variables": anom["anomalous_variables"]})
    drift_eval = drift_detector.evaluate() if len(drift_detector.window) >= drift_detector.window_size else {"status": "INSUFFICIENT_DATA"}
    drift_eval = _clean_nan(drift_eval)
    alert_manager.send("simulation", "Simulacion Anomala", "Se inyectaron " + str(payload.count) + " muestras anomalas. Drift: " + drift_eval.get("status", "N/A"), {"muestras": payload.count, "drift_status": drift_eval.get("status", "N/A")})
    return _clean_nan({"simulation": "anomaly", "count": payload.count, "results": results, "drift_after_simulation": drift_eval, "window_size": len(drift_detector.window)})

@app.post("/simulate/drift")
def simulate_drift(payload: SimulateDriftIn):
    cols = anomaly_detector.stats["columns"]
    if payload.shift_variable not in cols:
        raise HTTPException(status_code=400, detail="Variable desconocida: " + payload.shift_variable + ". Disponibles: " + str(cols))
    ranges = {
        "Alcohol": (10.0, 16.0), "Malic_Acid": (0.0, 6.0), "Ash": (1.0, 4.0),
        "Alcalinity_of_Ash": (10.0, 35.0), "Magnesium": (60.0, 180.0),
        "Total_Phenols": (0.5, 4.0), "Flavanoids": (0.0, 6.0),
        "Nonflavanoid_Phenols": (0.0, 1.0), "Proanthocyanins": (0.0, 4.0),
        "Color_Intensity": (1.0, 14.0), "Hue": (0.3, 2.0),
        "OD280_OD315": (1.0, 4.5), "Proline": (200.0, 1700.0),
    }
    results = []
    np.random.seed(123)
    for i in range(payload.count):
        csv_sample = {}
        for col in cols:
            mean = anomaly_detector.stats["means"][col]
            std = anomaly_detector.stats["stds"][col]
            if col == payload.shift_variable:
                val = mean + payload.shift_sigma * std + np.random.normal(0, std * 0.3)
            else:
                val = np.random.normal(mean, std * 0.5)
            low, high = ranges.get(col, (0.1, 9999))
            csv_sample[col] = float(np.clip(val, low, high))
        drift_detector.add_sample(csv_sample)
        results.append({"sample_id": i, payload.shift_variable: round(csv_sample[payload.shift_variable], 2)})
    drift_eval = drift_detector.evaluate() if len(drift_detector.window) >= drift_detector.window_size else {"status": "INSUFFICIENT_DATA"}
    drift_eval = _clean_nan(drift_eval)
    alert_manager.send("simulation", "Simulacion de Drift", "Se inyectaron " + str(payload.count) + " muestras con " + payload.shift_variable + " desplazada +" + str(payload.shift_sigma) + "sigma. Drift: " + drift_eval.get("status", "N/A"), {"variable": payload.shift_variable, "shift": payload.shift_sigma, "drift_status": drift_eval.get("status", "N/A")})
    return _clean_nan({"simulation": "drift", "count": payload.count, "shift_variable": payload.shift_variable, "shift_sigma": payload.shift_sigma, "drift_after_simulation": drift_eval, "window_size": len(drift_detector.window), "sample_preview": results[:5]})

# ------------------------------------------------------------------
# NUEVOS ENDPOINTS FASE 5: FEEDBACK LOOP
# ------------------------------------------------------------------

@app.post("/feedback")
def feedback(payload: FeedbackIn):
    """
    El enologo corrige una prediccion.
    Almacena: features + predicted_class + true_class + timestamp.
    Cuando se acumulan 10 feedbacks, se recomienda retraining.
    """
    sample_dict = payload.features.model_dump()
    csv_sample = _to_csv_names(sample_dict)
    result = feedback_store.add(csv_sample, payload.predicted_class, payload.true_class)
    summary = feedback_store.get_summary()
    
    # Alerta cuando se alcanzan 10 feedbacks (umbral de retraining)
    if summary["count"] == 10:
        alert_manager.send("feedback", "Retraining Recomendado", 
            "Se han acumulado 10 correcciones del enologo. Se recomienda ejecutar POST /retrain para reentrenar el modelo.",
            {"feedback_count": summary["count"], "error_rate": summary["error_rate"]}
        )
    
    return {
        "status": "feedback_stored",
        "feedback_count": summary["count"],
        "retrain_recommended": summary["retrain_recommended"],
        "predicted": payload.predicted_class,
        "true": payload.true_class,
        "match": payload.predicted_class == payload.true_class,
    }

@app.get("/feedback/status")
def feedback_status():
    """Estado del feedback acumulado."""
    return feedback_store.get_summary()

@app.post("/retrain")
def retrain():
    """
    Ejecuta el retraining del modelo mezclando datos originales + feedback.
    Requiere al menos 1 feedback acumulado.
    """
    summary = feedback_store.get_summary()
    if summary["count"] == 0:
        raise HTTPException(status_code=400, detail="No hay feedback acumulado. Envie correcciones via /feedback primero.")
    
    try:
        result = subprocess.run(
            ["python", "-m", "src.train"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/app"
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        
        # Recargar modelo automaticamente
        _load_model_internal()
        
        alert_manager.send("feedback", "Retraining Completado", 
            "El modelo ha sido reentrenado y recargado exitosamente.",
            {"feedback_used": summary["count"], "new_model": model_metadata["loaded_at"]}
        )
        
        return {
            "status": "retrained_and_reloaded",
            "model_metadata": model_metadata,
            "feedback_used": summary["count"],
            "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en retraining: {str(e)}")

@app.post("/reload")
def reload():
    """Recarga el modelo desde disco sin reiniciar el contenedor."""
    try:
        old_version = model_metadata.get("loaded_at", "unknown")
        _load_model_internal()
        return {
            "status": "reloaded",
            "previous_version": old_version,
            "new_version": model_metadata["loaded_at"],
            "model_type": model_metadata["model_type"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar modelo: {str(e)}")