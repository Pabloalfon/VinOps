FROM python:3.11-slim

WORKDIR /app

# 1. Copiar requirements.txt desde el PADRE (observabilidad/)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar el código del bonus desde bonus_shadowTesting/src/
COPY bonus_shadowTesting/src/ ./src/

# 3. Crear carpetas necesarias
RUN mkdir -p /app/models /app/shadow_logs /app/data /app/feedback

EXPOSE 8000

CMD ["uvicorn", "src.infer:app", "--host", "0.0.0.0", "--port", "8000"]