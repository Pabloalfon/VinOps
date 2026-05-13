"""VinOps Shadow Testing: Champion-Challenger.

El Champion responde al cliente. El Challenger predice en background
y se loguea para comparacion offline.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.shadow_tracker import ShadowTracker


CHAMPION_PATH = Path(os.getenv("CHAMPION_PATH", "/app/models/wine_model.pkl"))
CHALLENGER_PATH = Path(os.getenv("CHALLENGER_PATH", "/app/models/wine_model_challenger.pkl"))

MODEL_COLS = [
    "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium",
    "TotalPhenols", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins",
    "ColorIntensity", "Hue", "OD280_OD315", "Proline",
]


app = FastAPI(title="VinOps Shadow Testing", version="6.0.0")

champion = None
challenger = None
shadow_tracker = None


@app.on_event("startup")
def load_models() -> None:
    global champion, challenger, shadow_tracker
    
    # Champion siempre debe existir
    if not CHAMPION_PATH.exists():
        raise RuntimeError(f"Champion no encontrado: {CHAMPION_PATH}")
    
    champion = joblib.load(CHAMPION_PATH)
    print(f"[SHADOW] Champion cargado: {CHAMPION_PATH}")
    
    # Challenger es opcional (shadow solo si existe)
    if CHALLENGER_PATH.exists():
        challenger = joblib.load(CHALLENGER_PATH)
        print(f"[SHADOW] Challenger cargado: {CHALLENGER_PATH}")
    else:
        print(f"[SHADOW] Challenger NO encontrado. Shadow testing deshabilitado.")
    
    shadow_tracker = ShadowTracker()


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
    model: str  # "champion" o "challenger"
    shadow_comparison: Optional[Dict[str, Any]] = None


def _features_hash(features: Dict[str, float]) -> str:
    """Hash simple para trazabilidad."""
    return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()[:8]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "vinops-shadow-testing",
        "version": "6.0.0",
        "champion_loaded": champion is not None,
        "challenger_loaded": challenger is not None,
        "shadow_enabled": challenger is not None,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(features: WineFeatures):
    start_total = time.perf_counter()
    sample_dict = features.model_dump()
    sample_df = pd.DataFrame([sample_dict])[MODEL_COLS]
    
    # CHAMPION: predice y responde al cliente
    champ_start = time.perf_counter()
    champ_pred = int(champion.predict(sample_df.to_numpy())[0])
    champ_probas = champion.predict_proba(sample_df.to_numpy())[0]
    champ_conf = float(np.max(champ_probas))
    champ_latency = (time.perf_counter() - champ_start) * 1000
    
    # CHALLENGER: predice en shadow (no afecta la respuesta)
    shadow_comparison = None
    if challenger is not None:
        chall_start = time.perf_counter()
        chall_pred = int(challenger.predict(sample_df.to_numpy())[0])
        chall_probas = challenger.predict_proba(sample_df.to_numpy())[0]
        chall_conf = float(np.max(chall_probas))
        chall_latency = (time.perf_counter() - chall_start) * 1000
        
        # Loguear comparacion
        fh = _features_hash(sample_dict)
        shadow_tracker.log_comparison(
            champion_pred=champ_pred,
            challenger_pred=chall_pred,
            champion_conf=champ_conf,
            challenger_conf=chall_conf,
            latency_champion_ms=champ_latency,
            latency_challenger_ms=chall_latency,
            features_hash=fh,
        )
        
        shadow_comparison = {
            "challenger_prediction": chall_pred,
            "challenger_confidence": round(chall_conf, 4),
            "agreement": champ_pred == chall_pred,
            "confidence_delta": round(chall_conf - champ_conf, 4),
            "latency_challenger_ms": round(chall_latency, 2),
        }
    
    probas_dict = {f"class_{i+1}": round(float(p), 4) for i, p in enumerate(champ_probas)}
    
    return PredictionOut(
        prediction=champ_pred,
        class_label=f"Cultivar_{champ_pred}",
        probabilities=probas_dict,
        confidence=round(champ_conf, 4),
        model="champion",
        shadow_comparison=shadow_comparison,
    )


@app.get("/shadow/status")
def shadow_status():
    """Estadisticas acumuladas de Champion vs Challenger."""
    if challenger is None:
        raise HTTPException(status_code=503, detail="Challenger no cargado. Shadow testing no disponible.")
    
    summary = shadow_tracker.get_summary()
    return {
        "shadow_enabled": True,
        "champion_path": str(CHAMPION_PATH),
        "challenger_path": str(CHALLENGER_PATH),
        "statistics": summary,
    }


@app.post("/shadow/promote")
def shadow_promote():
    """Promueve el Challenger a Champion (copia el .pkl)."""
    if challenger is None:
        raise HTTPException(status_code=503, detail="No hay Challenger para promover.")
    
    if not CHALLENGER_PATH.exists():
        raise HTTPException(status_code=500, detail="Archivo Challenger no encontrado en disco.")
    
    # Copiar Challenger como Champion
    import shutil
    shutil.copy(CHALLENGER_PATH, CHAMPION_PATH)
    
    # Recargar Champion en memoria
    global champion
    champion = joblib.load(CHAMPION_PATH)
    
    return {
        "status": "promoted",
        "message": "Challenger promovido a Champion. El nuevo Champion esta activo.",
        "previous_champion": str(CHAMPION_PATH) + ".backup",
        "new_champion": str(CHAMPION_PATH),
    }


@app.get("/metrics")
def metrics():
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)