# Capítulo 5: Monitorización, Feedback Loop y Shadow Testing

> **Proyecto VinOps — Sistema de Clasificación de Cultivares de Vino**  
> **Asignatura:** Desarrollo e Integración de Servicios de Inteligencia Artificial  
> **Autor:** Pablo Alfonso López Fernández  
> **Fecha:** Mayo 2026

---

## Índice

1. [Introducción y objetivos](#1-introducción-y-objetivos)
2. [Arquitectura general del sistema de observabilidad](#2-arquitectura-general-del-sistema-de-observabilidad)
3. [Instrumentación y trazabilidad](#3-instrumentación-y-trazabilidad)
4. [Detección de deriva de datos (Data Drift)](#4-detección-de-deriva-de-datos-data-drift)
5. [Sistema de alertas y simulación de entornos anómalos](#5-sistema-de-alertas-y-simulación-de-entornos-anómalos)
6. [Visualización: Prometheus y Grafana](#6-visualización-prometheus-y-grafana)
7. [Feedback Loop y retraining semi-automático](#7-feedback-loop-y-retraining-semi-automático)
8. [Bonus: Shadow Testing (Champion-Challenger)](#8-bonus-shadow-testing-champion-challenger)
9. [Decisiones de diseño globales](#9-decisiones-de-diseño-globales)
10. [Evidencias y capturas requeridas](#10-evidencias-y-capturas-requeridas)
11. [Limitaciones y trabajo futuro](#11-limitaciones-y-trabajo-futuro)
12. [Conclusiones](#12-conclusiones)

---

## 1. Introducción y objetivos

El presente capítulo describe la implementación completa del sistema de monitorización, observabilidad y ciclo de vida MLOps para el proyecto VinOps. El objetivo principal es dotar al servicio de inferencia de capacidades que permitan detectar degradación del modelo en producción, alertar a los stakeholders y cerrar el ciclo de mejora continua mediante retraining con feedback real del enólogo.

### 1.1. Contexto del problema

VinOps opera en un entorno real donde la etiqueta del cultivar no está disponible en el momento de la predicción. El enólogo tarda horas o días en confirmarla mediante análisis organoléptico y químico. Esta característica impone una restricción crítica: **no es posible calcular métricas de rendimiento tradicionales (F1-Score, AUC, precisión) en tiempo real**. Por tanto, el sistema debe detectar degradación mediante métricas proxy y acumular ground truth para recalcular el rendimiento periódicamente.

### 1.2. Requisitos del profesorado

Durante la Sesión 13 se establecieron los siguientes requisitos mínimos:

| Requisito | Descripción |
|-----------|-------------|
| Observabilidad | Registrar métricas operativas (CPU, RAM, peticiones/s, latencia) y métricas del modelo |
| Detección de drift | Detectar deriva de datos y de conceptos en las muestras de entrada |
| Sistema de alertas | Al menos una alerta operativa y una de modelo, con simulación de entorno anómalo |
| Feedback Loop | Reentrenamiento simple con correcciones del enólogo |
| Bonus | A/B Testing o Shadow Testing |

### 1.3. Decisiones estratégicas previas

Antes de la implementación, se tomaron las siguientes decisiones arquitectónicas:

- **Separación en carpeta `observabilidad/`**: El código del Hito 5 se desacopló del Hito 4 para no contaminar el despliegue base, facilitando la revisión independiente.
- **Stack técnico ligero**: Uso de `scipy` en lugar de frameworks pesados (Alibi Detect, NannyML) para mantener el contenedor Docker manejable en un entorno académico.
- **Métricas proxy vs. métricas reales**: Se adoptó una estrategia de dos niveles —proxy online (sin ground truth) y métricas offline (con feedback acumulado)— justificada por la indisponibilidad de etiquetas en tiempo real.

---

## 2. Arquitectura general del sistema de observabilidad

### 2.1. Diagrama de componentes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTE                                         │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │ POST /predict
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API FASTAPI (Puerto 8001)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ /predict    │  │ /feedback    │  │ /retrain    │  │ /shadow/*       │   │
│  │ (inferencia)│  │ (corrección) │  │ (retraining)│  │ (bonus)         │   │
│  └──────┬──────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
│         │                                                                    │
│  ┌──────▼──────────────────────────────────────────────────────────────┐   │
│  │  INSTRUMENTACIÓN                                                    │   │
│  │  • Métricas Prometheus (CPU, RAM, latencia, confianza, entropía)   │   │
│  │  • Logging JSONL (trazabilidad completa)                            │   │
│  │  • Detección de anomalías ±3σ (Hito 2)                              │   │
│  │  • Detección de drift (KS + PSI + Wasserstein)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────┬──────────────────────────┬────────────────────────────┘
                      │                          │
          ┌───────────▼──────────┐   ┌──────────▼──────────┐
          │  Prometheus (9090)   │   │  Logs JSONL         │
          │  Scraping cada 5s    │   │  /app/logs/         │
          └───────────┬──────────┘   └─────────────────────┘
                      │
          ┌───────────▼──────────┐
          │  Grafana (3000)      │
          │  Dashboard 8 paneles │
          └──────────────────────┘
```

### 2.2. Flujo de datos

1. El cliente envía una muestra a `/predict`.
2. La API instrumenta la petición: contadores, histogramas, gauges.
3. El modelo predice y se calculan métricas proxy (confianza, entropía).
4. Se detectan anomalías (±3σ) y se evalúa drift si la ventana está llena.
5. Se escribe el log JSONL y se actualizan métricas Prometheus.
6. Si hay drift o métricas críticas, se envía alerta a Telegram.
7. El enólogo corrige vía `/feedback`, acumulando ground truth.
8. Al llegar a 10 feedbacks, se recomienda retraining.
9. `/retrain` mezcla datos originales + feedback, genera nuevo modelo.
10. `/reload` recarga el modelo sin reiniciar el contenedor.

---

## 3. Instrumentación y trazabilidad

### 3.1. Métricas operativas

Siguiendo la taxonomía del profesorado, se instrumentaron métricas operativas estándar de cualquier sistema software:

| Métrica | Tipo Prometheus | Descripción | Implementación |
|---------|----------------|-------------|----------------|
| `vinops_requests_total` | Counter | Peticiones HTTP por método, endpoint y código de estado | Middleware FastAPI |
| `vinops_request_latency_seconds` | Histogram | Latencia de peticiones a `/predict` | Middleware FastAPI |
| `vinops_cpu_usage_percent` | Gauge | Uso de CPU del proceso | `psutil.Process().cpu_percent()` |
| `vinops_memory_usage_bytes` | Gauge | Consumo de RAM residente | `psutil.Process().memory_info().rss` |

**Justificación de la elección de Prometheus:**

Prometheus es el estándar de facto en monitorización de sistemas cloud-native. Se eligió porque:
- Es un proyecto CNCF (Cloud Native Computing Foundation) con amplia adopción.
- El formato de métricas es texto plano, fácil de depurar.
- La integración con Grafana es nativa y no requiere conectores adicionales.
- El `prometheus_client` de Python es ligero y no añade dependencias pesadas.

**Alternativas descartadas:**
- **StatsD + Graphite**: Requiere daemon adicional y configuración más compleja.
- **InfluxDB**: Base de datos time-series potente pero overkill para un proyecto académico.
- **CloudWatch / Azure Monitor**: Dependencia de proveedor cloud, no aplicable a despliegue local.

### 3.2. Métricas proxy del modelo

Dado que en producción no se dispone de ground truth en tiempo real, se implementaron métricas proxy que detectan degradación sin etiqueta real:

| Métrica | Tipo | Justificación técnica |
|---------|------|----------------------|
| `vinops_prediction_confidence` | Histogram | Random Forest devuelve probabilidades por clase. Si la confianza media baja respecto al período de entrenamiento, el modelo está viendo vinos en su "zona gris" (frontera entre clase 2 y 3, detectada en Hito 3). |
| `vinops_prediction_entropy` | Histogram | Mide la "confusión" del modelo entre las 3 clases. Entropía alta = distribución uniforme de probabilidades = el modelo no distingue bien. Crítico para el solapamiento clase 2–clase 3. |
| `vinops_predictions_by_class_total` | Counter | Comparar histograma de predicciones en ventana deslizante vs. distribución del training (33%/40%/27%). Si de repente predice 80% de clase 3, hay cambio en la línea de producción o degradación. |
| `vinops_anomaly_flags_total` | Counter | Reutiliza umbrales PCA ±3σ del Hito 2. Monitorizar qué % de muestras superan esos umbrales indica calidad de la entrada. |
| `vinops_low_confidence_total` | Counter | % de predicciones donde la probabilidad máxima < 0.70. Si sube, el modelo está inseguro. |

**¿Por qué no F1-Score, AUC ni precisión online?**

Estas métricas requieren comparar `y_pred` con `y_true`. En VinOps, `y_true` (la etiqueta real del cultivar confirmada por el enólogo) no está disponible en el momento de la predicción. El enólogo la confirma horas o días después. Por tanto, calcular F1 en tiempo real es imposible sin un mecanismo de feedback previo. Las métricas proxy detectan degradación *antes* de saber si el modelo acertó, actuando como sistema de alerta temprana.

### 3.3. Detección de anomalías en inferencia

Se reutilizó la metodología del Hito 2 adaptándola a tiempo real. Para cada muestra entrante se calcula el z-score de cada variable respecto a la media y desviación del dataset de entrenamiento:

```
z_j = (x_j - x̄_j) / σ_j,    j = 1,...,13
```

Si |z_j| > 3 para cualquier variable, la muestra se marca como anómala. Esta información se incluye tanto en la respuesta JSON de `/predict` como en el log estructurado, permitiendo al enólogo revisar las variables específicas que desvían del perfil esperado.

**Justificación del umbral ±3σ:**

El umbral de 3 desviaciones estándar captura aproximadamente el 99.7% de los datos en una distribución normal. Fue establecido en el Hito 2 mediante análisis PCA y se mantiene para garantizar coherencia metodológica entre el análisis exploratorio y la monitorización operativa.

### 3.4. Logging estructurado

Cada predicción genera una línea JSON en `logs/vinops_predictions.jsonl` con:

- `timestamp`: instante ISO 8601 de la predicción.
- `input`: vector completo de las 13 variables fisicoquímicas.
- `prediction` y `probabilities`: clase predicha y distribución de probabilidades.
- `confidence` y `entropy`: métricas proxy de calidad.
- `anomaly`: resultado del detector con z-scores por variable.
- `latency_ms`: tiempo de respuesta de la inferencia.

**Justificación del formato JSONL:**

JSONL (JSON Lines) permite append eficiente sin reescribir el archivo completo, a diferencia de JSON estándar que requiere cierre de array. Facilita el procesamiento posterior con herramientas como `jq`, `pandas.read_json(lines=True)` o pipelines de ELK.

### 3.5. Endpoints del servicio monitorizado

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio, metadatos del modelo, resumen de monitorización y uso de CPU/RAM |
| `/predict` | POST | Predicción + métricas proxy + detección de anomalías + logging |
| `/metrics` | GET | Métricas Prometheus en formato text/plain |
| `/monitor` | GET | Resumen agregado: predicciones totales, anomalías, confianza media, distribución de clases |

---

## 4. Detección de deriva de datos (Data Drift)

### 4.1. Fundamento teórico

El *data drift* o deriva de datos se define como el cambio en la distribución de las variables de entrada P(**X**) entre el entrenamiento y la operación. A diferencia del *concept drift* —que afecta la relación P(Y|**X**)—, el data drift es detectable sin *ground truth*, lo cual lo hace especialmente relevante para VinOps.

Se distinguen tres niveles de severidad:

| Nivel | Condición | Acción recomendada |
|-------|-----------|-------------------|
| Drift leve | 0.1 < PSI < 0.25 | Monitorizar, no requiere acción inmediata |
| Drift moderado | PSI > 0.25 o KS_p < 0.05 en 1-2 variables | Revisar proceso de producción |
| Drift severo | KS_p < 0.05 en ≥3 variables o W > 2.0 | Reentrenamiento recomendado |

**Justificación de la elección de covariate shift:**

El profesor recomendó explícitamente enfocarse en la deriva de entrada: *"yo recomiendo que no os compliquéis y vayáis a la deriva de entrada"*. Se seleccionó el covariate shift porque:
1. Es el más frecuente en el dominio vitivinícola (cambios de cosecha, zona, clima).
2. No requiere ground truth para su detección.
3. Es el más sencillo de implementar y explicar.

### 4.2. Técnicas implementadas

Se implementaron tres técnicas complementarias, todas basadas en `scipy` para mantener el contenedor ligero:

| Técnica | Tipo | Umbrales | Justificación |
|---------|------|----------|---------------|
| **Kolmogorov-Smirnov (KS)** | Univariado | p < 0.05 por variable | Prueba no paramétrica estándar para comparar distribuciones continuas. No asume normalidad. |
| **Population Stability Index (PSI)** | Univariado | PSI > 0.25 (alerta), > 0.1 (monitor) | Métrica estándar en scoring y riesgo crediticio; fácil interpretación por umbrales. |
| **Wasserstein (media estandarizada)** | Multivariado | W > 2.0 (alerta), > 1.0 (monitor) | Detecta cambios conjuntos que las técnicas univariadas podrían omitir por efecto de compensación. |

**¿Por qué no Alibi Detect ni NannyML?**

Aunque son herramientas de referencia del estado del arte, se optó por implementación propia con `scipy` porque:
- Alibi Detect añade ~500 MB de dependencias (TensorFlow, PyTorch opcionales).
- NannyML requiere configuración de Reference/Analysis datasets formal.
- En un entorno académico, la implementación propia demuestra comprensión profunda de los fundamentos estadísticos.
- El contenedor se mantiene < 1 GB, facilitando el despliegue en máquinas del laboratorio.

En producción real se migraría a dichas herramientas, como se documenta en la memoria.

### 4.3. Ventana deslizante y evaluación periódica

Dado el volumen reducido del dataset (178 instancias), se configuró una ventana de **20 muestras** con evaluación cada **10 predicciones**. Esto equilibra:

- **Sensibilidad:** 20 muestras son suficientes para estimar 13 variables.
- **Latencia:** Evaluar cada 10 predicciones evita sobrecarga computacional.
- **Memoria:** La ventana se mantiene en memoria (deque) sin persistencia externa.

**Justificación del tamaño de ventana:**

Con 13 variables y 20 muestras, se tiene una razón de ~1.5 muestras por variable, suficiente para una estimación rudimentaria de la distribución. Ventanas mayores (50-100) serían más robustas estadísticamente pero introducirían latencia excesiva en la detección. Ventanas menores (5-10) serían demasiado ruidosas.

### 4.4. Regla de decisión global

El sistema combina las tres técnicas mediante una regla jerárquica:

```
DRIFT_DETECTED  si  N_KS ≥ 3  OR  N_PSI ≥ 2  OR  W > 2.0
MONITOR         si  N_KS ≥ 1  OR  N_PSI ≥ 1  OR  W > 1.0
OK              en otro caso
```

Donde N_KS es el número de variables con p < 0.05 y N_PSI el número de variables con PSI > 0.25.

**Justificación de la regla:**

- Requerir ≥3 variables en KS evita falsos positivos por fluctuación natural de una sola variable.
- PSI > 0.25 en ≥2 variables indica cambio sistémico, no puntual.
- Wasserstein > 2.0 captura desplazamientos conjuntos que KS/PSI podrían perder por compensación entre variables.

---

## 5. Sistema de alertas y simulación de entornos anómalos

### 5.1. Canal de alertas: Telegram Bot

Se seleccionó Telegram como canal de notificaciones por las siguientes razones:

| Criterio | Telegram | Correo (SMTP) | Teams (Webhook) |
|----------|----------|---------------|-----------------|
| Complejidad técnica | 🟢 Baja (HTTP + token) | 🔴 Alta (servidor SMTP, credenciales, puertos) | 🟡 Media (conector entrante, permisos de tenant) |
| Demo en vivo | 🟢 Sí, el tribunal ve el chat | 🟡 No visual | 🟡 Requiere pantalla aparte |
| Infraestructura | 🟢 Ninguna propia | 🔴 Servidor SMTP propio o externo | 🔴 Tenant de Microsoft |
| Tiempo de setup | 🟢 2 minutos | 🔴 15-30 minutos | 🟡 10-15 minutos |

**Proceso de configuración:**
1. Contactar a @BotFather en Telegram → `/newbot` → nombre `VinOpsAlertas`.
2. Obtener token de 46 caracteres.
3. Crear grupo con compañeros y profesor, añadir el bot.
4. Enviar mensaje en el grupo, consultar `getUpdates` para obtener `chat_id`.
5. Configurar token y chat_id en archivo `.env` (nunca en el código ni en Git).

**Seguridad:** El token se gestiona vía variables de entorno, con `.env` en `.gitignore`. El código usa `os.getenv()` para leerlo. Nunca está hardcodeado.

### 5.2. Throttling

Para evitar spam en escenarios de degradación continua, se implementó un mecanismo de *throttling* de **60 segundos por tipo de alerta**. Esto garantiza que el canal no se sature si el sistema detecta drift persistente durante varios minutos.

**Justificación del umbral de 60 segundos:**

- 30 segundos sería demasiado agresivo (podría perderse información relevante entre alertas).
- 5 minutos sería demasiado laxo (el enólogo no se enteraría de una degradación rápida).
- 60 segundos permite recibir alertas informativas sin saturar el grupo.

### 5.3. Tipos de alerta implementados

| Tipo | Trigger | Clase | Mensaje típico |
|------|---------|-------|----------------|
| Drift | N_KS ≥ 3 o N_PSI ≥ 2 o W > 2.0 | Modelo | "Deriva detectada en 5 variables. Recomendación: RETRAIN" |
| Anomalía | Tasa de anomalías > 25% en ventana | Modelo | "Tasa de anomalías elevada: 30% (7/20)" |
| Confianza baja | Confianza media < 0.65 en ventana | Modelo | "Confianza media baja: 0.58" |
| Operativa | CPU > 80% o RAM > 300 MB | Operativa | "CPU al 85%, umbral 80%" |
| Simulación | Endpoint `/simulate/*` invocado | Test | "Simulación de drift completada: 20 muestras" |

### 5.4. Simulación de entorno anómalo

Se implementaron dos endpoints de simulación para validar el sistema sin datos reales de producción:

**`/simulate/anomaly`:**
Genera N muestras con valores extremos (±2.5σ respecto a la media de referencia) pero dentro de los rangos físicos del vino. Permite verificar que el detector de anomalías y el sistema de alertas responden correctamente.

**`/simulate/drift`:**
Genera N muestras donde una variable específica está desplazada +kσ respecto a la distribución de entrenamiento, manteniendo el resto de variables normales. Simula un cambio real en la línea de producción (ej. uva de otra zona con mayor contenido alcohólico).

**Justificación de la simulación:**

En un entorno académico no se dispone de un flujo continuo de datos reales. La simulación permite:
- Validar que el detector de drift funciona con datos controlados.
- Demostrar al tribunal el sistema completo sin depender de tráfico externo.
- Reproducir escenarios específicos (ej. "¿qué pasa si el alcohol sube 3σ?").

---

## 6. Visualización: Prometheus y Grafana

### 6.1. Arquitectura de tres capas

1. **Capa de emisión:** La API expone métricas en formato Prometheus mediante `GET /metrics`.
2. **Capa de recolección:** Prometheus (puerto 9090) scrapea las métricas cada 5 segundos.
3. **Capa de visualización:** Grafana (puerto 3000) consume los datos y los presenta en dashboard preconfigurado.

### 6.2. Dashboard de Grafana

El dashboard "VinOps - Monitorización" incluye 8 paneles:

| Panel | Query Prometheus | Descripción |
|-------|-----------------|-------------|
| CPU Usage | `vinops_cpu_usage_percent` | Uso de CPU del proceso en tiempo real |
| Memory Usage | `vinops_memory_usage_bytes` | RAM consumida |
| Requests per Second | `rate(vinops_requests_total[1m])` | Throughput de la API |
| Request Latency | `histogram_quantile(0.95, ...)` | Latencia p50 y p95 en ms |
| Avg Model Confidence | `avg(vinops_prediction_confidence)` | Confianza media del modelo |
| Predictions by Class | `rate(vinops_predictions_by_class_total)` | Distribución de cultivares predichos |
| Anomalies & Low Confidence | `rate(vinops_anomaly_flags_total)` | Tasa de anomalías detectadas |
| Prediction Entropy | `avg(vinops_prediction_entropy)` | Entropía media (confusión del modelo) |

**Nota sobre CPU 0%:**

El modelo Random Forest es extremadamente ligero (178 muestras, 13 features). Una predicción consume microsegundos de CPU. El gauge de Prometheus muestrea el uso en instantes discretos, por lo que el promedio aparece como ~0%. Esto es **correcto y esperado**, no un error. Para observar CPU > 0% se requeriría saturar la API con cientos de peticiones por segundo, lo cual no es realista con este dataset.

---

## 7. Feedback Loop y retraining semi-automático

### 7.1. Arquitectura del ciclo de vida

```
Enólogo corrige → POST /feedback → CSV acumulado
                                          │
                                          ▼ (≥10 feedbacks)
                                    POST /retrain
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              wine_clean.csv       feedback.csv          wine_model.pkl
              (originales)         (correcciones)        (nuevo modelo)
                    │                     │                     │
                    └─────────────────────┘                     │
                              │                                 │
                              ▼                                 ▼
                         Dataset combinado                 POST /reload
                         (originales + feedback×2)         (sin reinicio)
```

### 7.2. Trigger de retraining: ¿por qué 10 feedbacks y no 10 drifts?

El retraining supervisado requiere **ground truth** (etiqueta real del cultivar). Un drift detecta cambio en la distribución de entrada, pero **no proporciona la etiqueta correcta** de las nuevas muestras. Sin saber si una muestra desplazada pertenece a la clase 1, 2 o 3, no es posible reentrenar el clasificador.

> *"NannyML te dice cuál va a ser la mejora o el decremento en la performance de tu modelo sin ground truth"* — Profesor, Sesión 13.

Las métricas proxy detectan degradación sin ground truth, pero el retraining efectivo requiere etiquetas reales. Por eso, el trigger es el feedback del enólogo, no el drift.

**Relación drift-feedback:**
- El drift detectado genera una alerta que notifica al enólogo.
- El enólogo, al revisar las muestras de esa ventana, proporciona correcciones.
- Cuando se acumulan 10 correcciones, el sistema recomienda retraining.

### 7.3. Almacenamiento en CSV

Las correcciones se almacenan en `feedback/feedback.csv` por las siguientes razones:

| Criterio | CSV | SQLite |
|----------|-----|--------|
| Visibilidad | Texto plano, abrible con Excel/Notepad | Binario, requiere herramientas |
| Dependencias | Ninguna | sqlite3 (aunque esté en stdlib) |
| Portabilidad | Copiar archivo es suficiente | Requiere dump |
| Volumen esperado | < 10.000 filas (decenas/mes) | Overkill |
| Transacciones | No necesarias (append-only) | Sí, pero no se aprovechan |

### 7.4. Mezcla de datos originales + feedback

El dataset Wine tiene solo **178 muestras**. Si se reentrenara únicamente con 10 feedbacks:

1. **Overfitting severo:** El modelo memorizaría las 10 muestras y perdería la generalización.
2. **Desbalanceo:** Las clases del feedback podrían no reflejar la distribución real.
3. **Catastrophic forgetting:** El modelo "olvidaría" los patrones aprendidos originalmente.

**Fórmula del dataset combinado:**
```
Dataset_final = Original (178) + Feedback × 2 (10 × 2 = 20) = 198 muestras
```

El feedback duplicado (×2) otorga mayor peso relativo a las correcciones del enólogo sin llegar a dominar el dataset. Con 20 muestras de feedback sobre 198 totales, el peso es ~10%, suficiente para ajustar fronteras sin destruir el conocimiento previo.

### 7.5. Recarga sin reinicio del contenedor

Tras el retraining, el modelo se recarga en memoria mediante `POST /reload` sin reiniciar Docker:

| Aspecto | Reinicio contenedor | Recarga en caliente |
|---------|---------------------|---------------------|
| Disponibilidad | ❌ Interrupción de servicio | ✅ Cero downtime |
| Estado de Prometheus | ❌ Métricas reseteadas | ✅ Preservadas |
| Ventana de drift | ❌ Perdida | ✅ Mantenida |
| Logs acumulados | ❌ Volúmenes re-montados | ✅ Intactos |

### 7.6. Resultados del retraining

| Métrica | Valor |
|---------|-------|
| Dataset original | 178 muestras |
| Feedback acumulado | 10 muestras |
| Feedback duplicado (×2) | 20 muestras |
| Dataset combinado | 198 muestras |
| F1-Score (macro) | 0.8757 |
| Balanced Accuracy | 0.8767 |

---

## 8. Bonus: Shadow Testing (Champion-Challenger)

### 8.1. Concepto y terminología

Siguiendo la terminología del profesorado, se implementó **Shadow Testing** (también conocido como *Champion-Challenger*):

- **Champion:** Modelo actual en producción. Es el único que responde al cliente.
- **Challenger:** Modelo candidato a reemplazar. Predice en *background* sobre las mismas muestras.

> *"Shadow testing es lo mismo que Champion Challenger... resuelve el problema del reparto de recursos, no que no tengamos que duplicar los recursos"* — Profesor, Sesión 13.

### 8.2. Arquitectura del shadow testing

```
Cliente → POST /predict
              │
              ├──► Champion predice → Respuesta al cliente
              │
              └──► Challenger predice (shadow) → ShadowTracker loguea
                          │
                          ▼
                   CSV de comparaciones
                          │
                          ▼
                   GET /shadow/status
                   POST /shadow/promote
```

### 8.3. Implementación

El Challenger recibe una **copia** de la petición, no parte del tráfico. Ambos modelos residen en el mismo contenedor, por lo que no se duplican recursos de infraestructura. La comparación se acumula en CSV con:

- Predicción de Champion y Challenger
- Confianza de ambos
- ¿Coinciden? (agreement)
- Delta de confianza
- Latencia de ambos

### 8.4. Promoción del Challenger

Cuando el operador decide que el Challenger es superior (o para la demo), `POST /shadow/promote` copia el `.pkl` del Challenger sobre el Champion y recarga el modelo en memoria, todo sin reiniciar el contenedor.

---

## 9. Decisiones de diseño globales

### 9.1. Resumen de decisiones clave

| Decisión | Alternativa descartada | Justificación |
|----------|----------------------|---------------|
| Métricas proxy (confianza, entropía) | F1/AUC online | Sin ground truth en tiempo real |
| Implementación propia con scipy | Alibi Detect / NannyML | Contenedor ligero, comprensión profunda |
| CSV para feedback | SQLite / PostgreSQL | Visibilidad, simplicidad, volumen reducido |
| Mezcla originales + feedback×2 | Solo feedback | Evita overfitting en dataset pequeño (178) |
| Trigger: 10 feedbacks | Trigger: 10 drifts | El drift no proporciona ground truth para retraining |
| Telegram | Correo / Teams | Simplicidad, demo en vivo, sin infraestructura propia |
| Recarga sin reinicio | Docker restart | Cero downtime, métricas preservadas |
| Shadow Testing | A/B Testing | A/B requiere división de público sin sesgar; Shadow es más seguro |

### 9.2. Implicaciones de las decisiones

- **Métricas proxy:** Permiten detectar degradación inmediata, pero con falsos positivos posibles. El enólogo debe validar las alertas.
- **CSV:** Facilita la depuración académica pero no escala a millones de registros. En producción se migraría a base de datos.
- **Mezcla con duplicado:** Garantiza robustez del modelo pero diluye el impacto de cada corrección individual.
- **Shadow Testing:** Permite validar modelos nuevos sin riesgo para el cliente, pero duplica la latencia de inferencia (aunque el Challenger no bloquea la respuesta).

---

## 10. Evidencias y capturas requeridas

### 10.1. Capturas obligatorias para la memoria

| # | Descripción | Dónde obtener |
|---|-------------|---------------|
| 1 | **Estructura de carpetas del proyecto** | Explorador de archivos, raíz de `observabilidad/` |
| 2 | **Terminal con `docker compose up --build` levantando** | Consola PowerShell durante el build |
| 3 | **Health check (`/health`) mostrando servicio activo** | Navegador o terminal, `http://localhost:8001/health` |
| 4 | **Predicción normal (`/predict`) con confianza alta** | Terminal con Invoke-RestMethod o navegador |
| 5 | **Predicción anómoma (`/predict`) con variables marcadas** | Misma muestra con valores extremos |
| 6 | **Endpoint `/metrics` de Prometheus** | Navegador, `http://localhost:9090/targets` mostrando **UP** |
| 7 | **Dashboard de Grafana con datos** | Navegador, `http://localhost:3000`, paneles poblados |
| 8 | **Alerta de drift en Telegram** | Captura del grupo de Telegram |
| 9 | **Alerta de simulación anómala en Telegram** | Captura del grupo de Telegram |
| 10 | **Estado de drift (`/drift`) mostrando DRIFT_DETECTED** | Terminal o navegador tras simulación |
| 11 | **Feedback acumulado (`/feedback/status`) con count=10** | Terminal tras enviar 10 feedbacks |
| 12 | **Retraining completado con métricas** | Terminal, output de `/retrain` |
| 13 | **Recarga del modelo (`/reload`) con nuevo timestamp** | Terminal, comparación de versiones |
| 14 | **Health del bonus (`/health` en puerto 8002)** | Terminal, `http://localhost:8002/health` |
| 15 | **Predicción con shadow comparison** | Terminal, `http://localhost:8002/predict` |
| 16 | **Estadísticas shadow (`/shadow/status`)** | Terminal con acuerdo y latencias |
| 17 | **Promoción del Challenger (`/shadow/promote`)** | Terminal, mensaje de éxito |

### 10.2. Capturas recomendadas adicionales

- Logs JSONL generados (`logs/vinops_predictions.jsonl`)
- CSV de feedback (`feedback/feedback.csv`)
- CSV de comparaciones shadow (`shadow_logs/comparisons.csv`)
- Panel de Grafana mostrando "Requests per Second" con tráfico

---

## 11. Limitaciones y trabajo futuro

### 11.1. Limitaciones actuales

1. **Sin versionado de modelos:** El nuevo `.pkl` sobrescribe el anterior. En producción se usaría MLflow.
2. **Sin rollback automático:** Si el retraining empeora el modelo, no hay mecanismo para revertir.
3. **Feedback no archivado:** Tras el retraining, el feedback sigue en el CSV. Debería marcarse como "usado".
4. **Challenger idéntico en demo:** Para la demo se usó el mismo modelo como Champion y Challenger. En producción real, el Challenger sería el modelo reentrenado con feedback.
5. **Sin auto-retraining:** El retraining requiere llamada manual a `/retrain`. Podría automatizarse.

### 11.2. Trabajo futuro

- **Auto-retraining:** Ejecutar retraining automáticamente cuando `feedback.count >= 10`.
- **NannyML integration:** Sustituir el detector propio por NannyML para estimar degradación de F1 sin ground truth.
- **MLflow Tracking:** Registrar cada retraining como run con métricas, parámetros y artefactos.
- **Canary deployment:** Desplegar nuevo modelo al 5% del tráfico antes de promoción completa.

---

## 12. Conclusiones

El Hito 5 ha implementado un sistema de observabilidad y ciclo de vida MLOps completo para VinOps, demostrando:

1. **Observabilidad:** Métricas operativas y proxy instrumentadas con Prometheus, visualizadas en Grafana.
2. **Detección proactiva:** Drift detectado mediante KS, PSI y Wasserstein antes de que el modelo falle.
3. **Alertas efectivas:** Notificaciones en Telegram con throttling, evitando spam.
4. **Ciclo cerrado:** El enólogo corrige, el sistema acumula, reentrena y despliega sin downtime.
5. **Validación segura:** Shadow Testing (Champion-Challenger) permite comparar modelos sin afectar al cliente.

La decisión clave —**mezclar datos originales + feedback duplicado**— resuelve el problema fundamental del dataset pequeño (178 muestras), evitando el overfitting y el catastrophic forgetting. El umbral de 10 feedbacks equilibra reactividad operativa con robustez estadística, mientras que las métricas proxy (confianza, entropía) actúan como sistema de alerta temprana en ausencia de ground truth inmediato.

---

*Documentación generada para el Proyecto DISIA — VinOps.*  
*Hito 5: Monitorización, Feedback Loop y Shadow Testing.*  
*Mayo 2026*
