# Fase 5: Feedback Loop y Retraining Semi-Automático

## 1. Introducción

El **Feedback Loop** constituye el cierre del ciclo de vida MLOps en VinOps. Su objetivo es permitir que el enólogo corrija las predicciones del modelo en producción, acumular esas correcciones como *ground truth* y utilizarlas para reentrenar periódicamente el modelo, adaptándolo a la realidad operativa sin intervención manual del equipo de desarrollo.

> *"El mínimo es que hagáis eso... llamáis a vuestra API del reentrenamiento"* — Profesor, Sesión 13.

---

## 2. Arquitectura del Sistema

```
┌─────────────────┐     POST /feedback      ┌─────────────────┐
│   Enólogo       │ ──────────────────────► │  FeedbackStore  │
│  (corrección)   │  {features, predicted,  │   (CSV local)   │
│                 │   true_class}           │                 │
└─────────────────┘                         └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  Acumulación    │
                                            │  ≥10 feedbacks  │
                                            └────────┬────────┘
                                                     │
                                                     ▼ trigger
┌─────────────────┐     POST /retrain       ┌─────────────────┐
│   Operador      │ ──────────────────────► │   src/train.py  │
│  (o alerta)     │                         │  (retraining)   │
└─────────────────┘                         └────────┬────────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              │                      │                      │
                              ▼                      ▼                      ▼
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │ wine_clean.csv  │    │ feedback.csv    │    │ wine_model.pkl  │
                    │  (originales)   │ +  │ (correcciones)  │ →  │  (nuevo modelo) │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                             │
                                                                             ▼
                                                                   ┌─────────────────┐
                                                                   │  POST /reload   │
                                                                   │ (sin reinicio)  │
                                                                   └─────────────────┘
```

---

## 3. Decisiones de Diseño y Justificaciones

### 3.1. Trigger de retraining: cada 10 feedbacks (no cada 10 drifts)

**Decisión:** El retraining se dispara cuando se acumulan **10 correcciones del enólogo** vía `/feedback`, no cuando se detectan 10 drifts.

**Justificación técnica:**

El retraining supervisado requiere **ground truth** (etiqueta real del cultivar). Un drift de datos (`covariate shift`) indica que la distribución de las variables de entrada ha cambiado, pero **no proporciona la etiqueta correcta** de las nuevas muestras. Sin saber si una muestra desplazada pertenece a la clase 1, 2 o 3, no es posible reentrenar el clasificador.

> *"NannyML te dice cuál va a ser la mejora o el decremento en la performance de tu modelo sin ground truth"* — Profesor, Sesión 13.

Las métricas proxy (confianza, entropía, drift) detectan degradación *sin* ground truth, pero el **retraining efectivo requiere etiquetas reales**. Por eso, el trigger es el feedback del enólogo, no el drift.

**Relación drift-feedback:**
- El drift detectado genera una **alerta** que notifica al enólogo.
- El enólogo, al revisar las muestras de esa ventana, proporciona correcciones.
- Cuando se acumulan 10 correcciones, el sistema recomienda retraining.

---

### 3.2. Almacenamiento en CSV (no SQLite ni base de datos)

**Decisión:** Las correcciones se almacenan en un archivo CSV plano (`feedback/feedback.csv`).

**Justificación:**

| Criterio | CSV | SQLite | Justificación |
|----------|-----|--------|---------------|
| **Visibilidad** | ✅ Texto plano | ❌ Binario | El profesor y el equipo pueden abrir el CSV con Excel/Notepad y ver las correcciones en crudo. |
| **Dependencias** | ✅ Ninguna | ❌ `sqlite3` | Aunque sqlite3 está en la stdlib, CSV es más ligero y no requiere conexiones ni esquemas. |
| **Portabilidad** | ✅ Copiar archivo | ❌ Dump necesario | Para la entrega académica, un CSV es un artefacto autocontenido. |
| **Volumen esperado** | ✅ < 10.000 filas | ⚠️ Overkill | En producción real de un enólogo, el feedback es de decenas o cientos de muestras al mes, no millones. |
| **Transacciones** | ❌ No | ✅ Sí | No necesitamos ACID; el feedback es append-only. |

**Conclusión:** Para el volumen y el contexto académico de VinOps, CSV es la opción más simple, transparente y mantenible.

---

### 3.3. Mezcla de datos originales + feedback (no solo feedback)

**Decisión:** El retraining mezcla el dataset original (`wine_clean.csv`, 178 muestras) con el feedback acumulado, **duplicando el feedback ×2** para darle peso sin dominar.

**Justificación (dataset pequeño):**

El dataset Wine tiene solo **178 muestras**. Si reentrenáramos únicamente con 10 feedbacks:

1. **Overfitting severo:** El modelo memorizaría las 10 muestras y perdería la generalización aprendida de los 178 originales.
2. **Desbalanceo:** Las clases del feedback podrían no reflejar la distribución real (ej. el enólogo corrige más la clase 2 porque es la más difícil).
3. **Catastrophic forgetting:** El modelo "olvidaría" los patrones aprendidos en el entrenamiento original.

**Fórmula del dataset combinado:**

```
Dataset_final = Original (178) + Feedback × 2 (10 × 2 = 20)
              = 198 muestras
```

El feedback duplicado (×2) otorga **mayor peso relativo** a las correcciones del enólogo sin llegar a dominar el dataset. Con 20 muestras de feedback sobre 198 totales, el peso es ~10%, suficiente para ajustar fronteras sin destruir el conocimiento previo.

> *"Con pocos datos de feedback, el balanceo evita que las clases minoritarias se ignoren"* — Decisión documentada en `src/train.py`.

---

### 3.4. Recarga sin reinicio del contenedor

**Decisión:** Tras el retraining, el modelo se recarga en memoria mediante `POST /reload` sin reiniciar el contenedor Docker.

**Justificación:**

| Aspecto | Reinicio contenedor | Recarga en caliente |
|---------|---------------------|---------------------|
| **Disponibilidad** | ❌ Interrupción de servicio | ✅ Cero downtime |
| **Estado de Prometheus** | ❌ Métricas reseteadas | ✅ Preservadas |
| **Ventana de drift** | ❌ Perdida | ✅ Mantenida |
| **Logs acumulados** | ❌ Volúmenes re-montados | ✅ Intactos |
| **Complejidad** | ⚠️ Docker Compose restart | ✅ `joblib.load()` en Python |

La recarga en caliente (`_load_model_internal()`) lee el nuevo `.pkl` desde disco y actualiza la variable global `model` en el proceso Uvicorn. El endpoint `/health` refleja inmediatamente el nuevo `loaded_at`.

---

### 3.5. Umbral de 10 feedbacks

**Decisión:** El umbral para recomendar retraining es **10 correcciones**.

**Justificación:**

- **Estadístico:** Con 10 muestras y 3 clases, se garantiza al menos 2-3 muestras por clase (en promedio), suficiente para detectar un sesgo sistemático (ej. "el modelo confunde clase 2 con 3").
- **Operativo:** Un enólogo que corrige 10 vinos en una jornada de cata representa un volumen realista de trabajo.
- **Académico:** 10 es un número redondo, fácil de demostrar en la presentación sin saturar la demo.

---

## 4. Implementación Técnica

### 4.1. Endpoints del Feedback Loop

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/feedback` | `POST` | El enólogo envía una corrección: features, clase predicha y clase real. |
| `/feedback/status` | `GET` | Devuelve el estado del acumulador: count, error_rate, retrain_recommended. |
| `/retrain` | `POST` | Ejecuta `src/train.py`, mezcla datos, entrena nuevo modelo y recarga automáticamente. |
| `/reload` | `POST` | Recarga el modelo desde disco sin reiniciar el contenedor. |

### 4.2. Formato del CSV de feedback

```csv
Alcohol,Malic_Acid,Ash,Alcalinity_of_Ash,Magnesium,Total_Phenols,Flavanoids,Nonflavanoid_Phenols,Proanthocyanins,Color_Intensity,Hue,OD280_OD315,Proline,predicted_class,true_class,timestamp
13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050,1,2,2026-05-12T23:20:09
```

**13 features** en snake_case (coinciden con `wine_clean.csv`) + `predicted_class` + `true_class` + `timestamp`.

### 4.3. Robustez ante corrupción

El `FeedbackStore` implementa:
- **Detección de columnas corruptas:** Si el CSV existente no tiene las columnas esperadas, se recrea automáticamente.
- **Filtrado defensivo:** `get_summary()` solo calcula métricas si las columnas `predicted_class` y `true_class` existen.
- **Limpieza programática:** Método `clear()` para resetear el acumulador tras retraining exitoso (opcional).

### 4.4. Pipeline de retraining

```python
# src/train.py
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_features=2,          # mtry=2 (equivalente a R)
        random_state=42,
        class_weight="balanced", # Compensa desbalanceo con pocos datos
    )),
])
```

**Hiperparámetros mantenidos:**
- `n_estimators=200`: Mismo bosque que el modelo original.
- `max_features=2`: Equivalente a `mtry=2` en `caret::train()` del Hito 3.
- `class_weight="balanced"`: Crítico con datasets pequeños; evita que las clases minoritarias del feedback sean ignoradas.

---

## 5. Flujo de Trabajo (Demo)

### Paso 1: El enólogo recibe una predicción
```
POST /predict → {"prediction": 1, "confidence": 0.986}
```

### Paso 2: El enólogo corrige
```
POST /feedback
{
  "features": {"Alcohol": 13.2, "MalicAcid": 1.78, ...},
  "predicted_class": 1,
  "true_class": 2
}
→ {"status": "feedback_stored", "feedback_count": 1}
```

### Paso 3: Acumulación
```
GET /feedback/status → {"count": 10, "retrain_recommended": true, "error_rate": 0.4}
```

### Paso 4: Alerta a Telegram
> 🔔 **Retraining Recomendado** — Se han acumulado 10 correcciones del enólogo.

### Paso 5: Retraining
```
POST /retrain → {"status": "retrained_and_reloaded", "f1_score": 0.8757}
```

### Paso 6: Verificación
```
GET /health → {"model": {"loaded_at": "2026-05-12T23:20:12.089324"}}
```

### Paso 7: Predicción con modelo actualizado
```
POST /predict → {"prediction": 2, "confidence": 0.6431, "model_version": "2026-05-12T23:20:12.089324"}
```

---

## 6. Métricas de Validación

### Resultados del retraining (prueba real)

| Métrica | Valor |
|---------|-------|
| Dataset original | 178 muestras |
| Feedback acumulado | 10 muestras |
| Feedback duplicado (×2) | 20 muestras |
| **Dataset combinado** | **198 muestras** |
| **F1-Score (macro)** | **0.8757** |
| **Balanced Accuracy** | **0.8767** |

### Comparativa modelo original vs. reentrenado

| Aspecto | Modelo Original | Modelo Reentrenado |
|---------|-----------------|-------------------|
| F1-Score | ~0.98 (sobre training) | 0.8757 (sobre test) |
| Confianza típica | ~0.98 | ~0.64 (más honesta) |
| Comportamiento | Sobreajustado a originales | Generalizado con feedback |

> La caída de confianza es **deseable**: el modelo reentrenado es más conservador ante muestras que no vio en el entrenamiento original, lo cual es signo de mejor generalización.

---

## 7. Limitaciones y Trabajo Futuro

### Limitaciones actuales

1. **Sin versionado de modelos:** El nuevo `.pkl` sobrescribe el anterior. En producción real se usaría MLflow o un sistema de versionado (`wine_model_v1.pkl`, `wine_model_v2.pkl`).
2. **Sin rollback automático:** Si el retraining empeora el modelo (F1 baja drásticamente), no hay mecanismo para revertir al modelo anterior.
3. **Feedback no archivado:** Tras el retraining, el feedback sigue en el CSV. En producción debería archivarse o marcarse como "usado".
4. **Sin validación del enólogo:** El feedback se acepta sin verificar si el enólogo tiene autoridad (cualquiera con acceso a `/feedback` puede corregir).

### Trabajo futuro (bonus)

- **A/B Testing:** Desplegar modelo A (original) y modelo B (reentrenado) en paralelo, dividiendo el tráfico 50/50 y comparando métricas offline.
- **Shadow Testing:** El modelo nuevo recibe copia de las peticiones pero no responde al cliente; se comparan sus predicciones con el modelo activo.
- **Auto-retraining:** Eliminar el endpoint `/retrain` manual y ejecutar el retraining automáticamente cuando `feedback.count >= 10`.
- **NannyML integration:** Sustituir el detector de drift propio por NannyML para estimar la degradación de F1 sin ground truth.

---

## 8. Conclusiones

El Feedback Loop de VinOps demuestra un ciclo de vida MLOps completo:

1. **Observar:** Métricas proxy detectan degradación (Fase 1-2).
2. **Alertar:** Telegram notifica al enólogo (Fase 3).
3. **Corregir:** El enólogo proporciona ground truth vía `/feedback` (Fase 5).
4. **Reentrenar:** El modelo se ajusta con datos originales + feedback (Fase 5).
5. **Desplegar:** Recarga en caliente sin downtime (Fase 5).

La decisión clave —**mezclar originales + feedback duplicado**— resuelve el problema fundamental del dataset pequeño (178 muestras), evitando el overfitting y el catastrophic forgetting. El umbral de 10 feedbacks equilibra reactividad operativa con robustez estadística.

---

*Documentación generada para el Hito 5 del Proyecto DISIA — VinOps.*
*Fecha: 2026-05-13*
