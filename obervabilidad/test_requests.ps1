# ============================================
# test_requests.ps1
# Script de prueba para VinOps Observabilidad (PowerShell)
# Ejecutar desde la carpeta observabilidad/
# ============================================

$BaseUrl = "http://localhost:8001"

function Write-Header($title) {
    Write-Host ""
    Write-Host ("=== " + $title + " ===") -ForegroundColor Cyan
}

# 1. Health check
Write-Header "1. Health check"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 2. Prediccion NORMAL (Clase 1 tipica)
Write-Header "2. Prediccion NORMAL (Clase 1 tipica)"
$payloadNormal = @{
    Alcohol = 14.23; MalicAcid = 1.71; Ash = 2.43; AlcalinityAsh = 15.6
    Magnesium = 127; TotalPhenols = 2.80; Flavanoids = 3.06; NonflavanoidPhenols = 0.28
    Proanthocyanins = 2.29; ColorIntensity = 5.64; Hue = 1.04; OD280_OD315 = 3.92
    Proline = 1065
} | ConvertTo-Json -Compress

try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payloadNormal
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 3. Prediccion FRONTERA (valores intermedios)
Write-Header "3. Prediccion FRONTERA (valores intermedios)"
$payloadBorder = @{
    Alcohol = 12.5; MalicAcid = 2.5; Ash = 2.2; AlcalinityAsh = 20.0
    Magnesium = 95; TotalPhenols = 2.0; Flavanoids = 1.5; NonflavanoidPhenols = 0.35
    Proanthocyanins = 1.5; ColorIntensity = 4.0; Hue = 0.9; OD280_OD315 = 2.5
    Proline = 500
} | ConvertTo-Json -Compress

try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payloadBorder
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 4. Prediccion ANOMALA (valores extremos pero dentro de rangos)
Write-Header "4. Prediccion ANOMALA (valores extremos)"
$payloadAnomaly = @{
    Alcohol = 11.5; MalicAcid = 5.5; Ash = 3.2; AlcalinityAsh = 29.0
    Magnesium = 160; TotalPhenols = 0.5; Flavanoids = 0.5; NonflavanoidPhenols = 0.6
    Proanthocyanins = 3.5; ColorIntensity = 12.0; Hue = 0.5; OD280_OD315 = 1.3
    Proline = 1600
} | ConvertTo-Json -Compress

try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payloadAnomaly
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 5. Metricas Prometheus
Write-Header "5. Metricas Prometheus (primeras 40 lineas)"
try {
    $metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET
    ($metrics -split "`n") | Select-Object -First 40
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 6. Resumen Monitor
Write-Header "6. Resumen de monitorizacion"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/monitor" -Method GET
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 7. Logs generados en el contenedor
Write-Header "7. Logs generados en el contenedor"
try {
    docker exec vinops_inference_monitored ls -la /app/logs/
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 8. Ultima linea del log
Write-Header "8. Ultima linea del log"
try {
    docker exec vinops_inference_monitored tail -n 1 /app/logs/vinops_predictions.jsonl
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Pruebas completadas ===" -ForegroundColor Green
