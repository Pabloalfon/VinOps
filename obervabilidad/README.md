# VinOps – Observabilidad (Hito 5, Fase 1)

Sistema de instrumentación y trazabilidad para el servicio de inferencia VinOps.

## Estructura

```
observabilidad/
├── src/
│   ├── infer.py            # API FastAPI instrumentada
│   ├── monitoring.py       # Métricas Prometheus + logging JSON
│   └── anomaly_detector.py # Detección de anomalías ±3σ (Hito 2)
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── logs/                   # Volúmen de logs JSONL
├── models/                 # Copiar aquí wine_model.pkl
└── test_requests.sh        # Script de prueba
```

## Requisitos previos

1. Tener `wine_clean.csv` en la raíz del proyecto (se monta vía volumen).
2. Copiar tu modelo entrenado `wine_model.pkl` (del Hito 4) a `observabilidad/models/`:
   ```bash
   cp /ruta/a/tu/wine_model.pkl observabilidad/models/
   ```

## Despliegue

```bash
cd observabilidad
docker compose up --build
```

El servicio levanta en `http://localhost:8001`.

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio, metadatos del modelo, uso de CPU/RAM |
| `/predict` | POST | Predicción + métricas proxy + detección de anomalías |
| `/metrics` | GET | Métricas en formato Prometheus (text/plain) |
| `/monitor` | GET | Resumen agregado de predicciones, drift y confianza |

## Métricas implementadas

### Operativas (Prometheus)
- `vinops_requests_total` – Contador de peticiones (método, endpoint, status)
- `vinops_request_latency_seconds` – Histograma de latencia
- `vinops_cpu_usage_percent` – Uso de CPU del proceso
- `vinops_memory_usage_bytes` – Uso de RAM del proceso

### Proxy del modelo (Prometheus)
- `vinops_prediction_confidence` – Confianza máxima (probabilidad)
- `vinops_prediction_entropy` – Entropía de la distribución de clases
- `vinops_predictions_by_class_total` – Predicciones acumuladas por cultivar
- `vinops_anomaly_flags_total` – Muestras fuera de ±3σ
- `vinops_low_confidence_total` – Predicciones con confianza < 0.70

### Logging estructurado
Cada predicción genera una línea JSON en `logs/vinops_predictions.jsonl` con:
- timestamp, input completo, predicción, probabilidades, confianza, entropía
- resultado del detector de anomalías (z-scores por variable)
- latencia de la petición

## Pruebas

```bash
chmod +x test_requests.sh
./test_requests.sh
```

O manualmente:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Alcohol": 13.2, "MalicAcid": 1.78, "Ash": 2.14, "AlcalinityAsh": 11.2,
    "Magnesium": 100, "TotalPhenols": 2.65, "Flavanoids": 2.76,
    "NonflavanoidPhenols": 0.26, "Proanthocyanins": 1.28,
    "ColorIntensity": 4.38, "Hue": 1.05, "OD280_OD315": 3.40, "Proline": 1050
  }'
```

## Próximas fases

- **Fase 2:** Prometheus + Grafana para visualización de métricas.
- **Fase 3:** Feedback loop (`/feedback`) y retraining semi-automático.
- **Fase 4:** Integración con MLflow Tracking del Hito 4.
