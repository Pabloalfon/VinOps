# VinOps - Fase 4: Prometheus + Grafana

Visualizacion de metricas para el sistema VinOps.

## Archivos necesarios

```
obervabilidad/
├── docker-compose.yaml          (reemplazar por el nuevo)
├── prometheus.yml               (nuevo)
├── .env                         (ya lo tienes)
├── grafana/
│   ├── datasources/
│   │   └── datasource.yml       (nuevo)
│   └── dashboards/
│       ├── dashboard.yml        (nuevo)
│       └── vinops-dashboard.json (nuevo)
```

## Despliegue

### 1. Baja todo lo anterior

```bash
cd obervabilidad
docker compose down
```

### 2. Copia los nuevos archivos

- `docker-compose-prometheus.yaml` → renombrar a `docker-compose.yaml`
- `prometheus.yml` → raiz de `obervabilidad/`
- Carpeta `grafana/` → dentro de `obervabilidad/`

### 3. Levanta los servicios

```bash
docker compose up --build
```

Esto levanta 3 contenedores:
- `vinops_inference_monitored` (API en puerto 8001)
- `vinops_prometheus` (scraper en puerto 9090)
- `vinops_grafana` (dashboard en puerto 3000)

### 4. Accede a las interfaces

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| API VinOps | http://localhost:8001 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

### 5. Verifica en Prometheus

1. Abre http://localhost:9090
2. Ve a "Status" → "Targets"
3. Deberias ver `vinops` con estado **UP**

### 6. Verifica en Grafana

1. Abre http://localhost:3000
2. Login: `admin` / `admin`
3. Ve a "Dashboards" → "Browse"
4. Selecciona "VinOps - Monitorizacion"
5. Deberias ver los paneles con datos en tiempo real

### 7. Genera trafico para ver datos

```powershell
# En otra terminal, ejecuta las pruebas
powershell -ExecutionPolicy Bypass -File .	est_requests_fase23.ps1
```

Vuelve a Grafana y refresca (F5). Los paneles mostraran:
- Peticiones por segundo
- Latencia p50/p95
- Uso de CPU y RAM
- Confianza media del modelo
- Distribucion de clases predichas
- Tasa de anomalias
- Entropia de predicciones

## Paneles del Dashboard

| Panel | Metrica | Descripcion |
|-------|---------|-------------|
| CPU Usage | `vinops_cpu_usage_percent` | Uso de CPU del proceso |
| Memory Usage | `vinops_memory_usage_bytes` | RAM consumida |
| Requests per Second | `rate(vinops_requests_total[1m])` | Throughput de la API |
| Request Latency | `histogram_quantile(0.95, ...)` | Latencia p50 y p95 |
| Avg Model Confidence | `avg(vinops_prediction_confidence)` | Confianza media |
| Predictions by Class | `rate(vinops_predictions_by_class_total)` | Distribucion de clases |
| Anomalies & Low Confidence | `rate(vinops_anomaly_flags_total)` | Tasa de anomalias |
| Prediction Entropy | `avg(vinops_prediction_entropy)` | Entropia media |
