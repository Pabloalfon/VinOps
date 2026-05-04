import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar modelo al iniciar
model_path = os.getenv("MODEL_PATH", "/models/wine_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en: {model_path}. Ejecuta primero el entrenamiento.")

with open(model_path, "rb") as f:
    modelo = pickle.load(f)

print(f"Modelo cargado. Tipo: {type(modelo).__name__}")

app = FastAPI(
    title="VinOps API",
    description="Clasificación de vinos (clase 1, 2 o 3) basada en propiedades fisicoquímicas"
)

# Definir el esquema de entrada
class WineFeatures(BaseModel):
    Alcohol: float
    Malic_Acid: float
    Ash: float
    Alcalinity_of_Ash: float
    Magnesium: float
    Total_Phenols: float
    Flavanoids: float
    Nonflavanoid_Phenols: float
    Proanthocyanins: float
    Color_Intensity: float
    Hue: float
    OD280_OD315: float
    Proline: float

@app.get("/health")
def health():
    return {
        "status": "OK",
        "model": "Random Forest",
        "classes": [1, 2, 3]
    }

@app.post("/predict")
def predict(features: WineFeatures):
    df = pd.DataFrame([features.dict()])
    
    # Predecir
    pred = modelo.predict(df)[0]
    proba = modelo.predict_proba(df)[0]
    
    return {
        "prediccion": int(pred),
        "probabilidades": {
            "1": round(float(proba[0]), 4),
            "2": round(float(proba[1]), 4),
            "3": round(float(proba[2]), 4)
        },
        "status": 200
    }

@app.get("/metadata")
def metadata():
    return {
        "modelo": "Random Forest",
        "framework": "scikit-learn",
        "clases": [1, 2, 3],
        "num_features": 13,
        "features": list(modelo.feature_names_in_)
    }