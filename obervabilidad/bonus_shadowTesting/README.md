# Bonus: Shadow Testing (Champion-Challenger)

## Concepto
El modelo **Champion** (actual en produccion) responde al cliente.
El modelo **Challenger** (nuevo, reentrenado) predice en background
sobre las mismas muestras sin afectar la respuesta.

## Arquitectura

Cliente -> POST /predict -> Champion responde
-> Challenger predice (shadow)
-> ShadowTracker loguea comparacion


## Endpoints
- `POST /predict` -> Respuesta del Champion + comparacion shadow
- `GET /shadow/status` -> Estadisticas acumuladas
- `POST /shadow/promote` -> Challenger se convierte en Champion

## Como usar
1. Entrenar Challenger desde el Hito 5 (`/retrain`)
2. Copiar modelo a `bonus_shadowTesting/models/wine_model_challenger.pkl`
3. `docker compose up --build`
4. Enviar predicciones y consultar `/shadow/status`

## Terminologia
- **Champion**: Modelo actual en produccion (wine_model.pkl)
- **Challenger**: Modelo candidato a reemplazar (wine_model_challenger.pkl)
- **Shadow**: Prediccion en background sin afectar al cliente