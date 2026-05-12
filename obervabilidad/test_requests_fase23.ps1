# ============================================
# test_requests_fase23.ps1
# Script de prueba para VinOps Fase 2+3 (PowerShell)
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

# 2. Llenando ventana con 20 predicciones normales (con ligera variacion)
Write-Header "2. Llenando ventana con 20 predicciones normales"

# Valores base de una muestra tipica de Clase 1
$baseValues = @{
    Alcohol = 13.2; MalicAcid = 1.78; Ash = 2.14; AlcalinityAsh = 11.2
    Magnesium = 100; TotalPhenols = 2.65; Flavanoids = 2.76; NonflavanoidPhenols = 0.26
    Proanthocyanins = 1.28; ColorIntensity = 4.38; Hue = 1.05; OD280_OD315 = 3.40
    Proline = 1050
}

# Desviaciones tipicas para variacion (pequeno ruido gaussiano)
$stdValues = @{
    Alcohol = 0.8; MalicAcid = 0.5; Ash = 0.2; AlcalinityAsh = 2.0
    Magnesium = 10; TotalPhenols = 0.3; Flavanoids = 0.4; NonflavanoidPhenols = 0.05
    Proanthocyanins = 0.3; ColorIntensity = 0.8; Hue = 0.1; OD280_OD315 = 0.4
    Proline = 100
}

for ($i = 1; $i -le 20; $i++) {
    # Generar muestra con ruido gaussiano pequeno
    $sample = @{}
    foreach ($key in $baseValues.Keys) {
        $noise = Get-Random -Minimum -0.3 -Maximum 0.3
        $val = $baseValues[$key] + $noise * $stdValues[$key]
        $sample[$key] = [math]::Round($val, 2)
    }

    # Asegurar rangos Pydantic
    $sample["Alcohol"] = [math]::Max(10.0, [math]::Min(16.0, $sample["Alcohol"]))
    $sample["MalicAcid"] = [math]::Max(0.0, [math]::Min(6.0, $sample["MalicAcid"]))
    $sample["Ash"] = [math]::Max(1.0, [math]::Min(4.0, $sample["Ash"]))
    $sample["AlcalinityAsh"] = [math]::Max(10.0, [math]::Min(35.0, $sample["AlcalinityAsh"]))
    $sample["Magnesium"] = [math]::Max(60.0, [math]::Min(180.0, $sample["Magnesium"]))
    $sample["TotalPhenols"] = [math]::Max(0.5, [math]::Min(4.0, $sample["TotalPhenols"]))
    $sample["Flavanoids"] = [math]::Max(0.0, [math]::Min(6.0, $sample["Flavanoids"]))
    $sample["NonflavanoidPhenols"] = [math]::Max(0.0, [math]::Min(1.0, $sample["NonflavanoidPhenols"]))
    $sample["Proanthocyanins"] = [math]::Max(0.0, [math]::Min(4.0, $sample["Proanthocyanins"]))
    $sample["ColorIntensity"] = [math]::Max(1.0, [math]::Min(14.0, $sample["ColorIntensity"]))
    $sample["Hue"] = [math]::Max(0.3, [math]::Min(2.0, $sample["Hue"]))
    $sample["OD280_OD315"] = [math]::Max(1.0, [math]::Min(4.5, $sample["OD280_OD315"]))
    $sample["Proline"] = [math]::Max(200.0, [math]::Min(1700.0, $sample["Proline"]))

    $payload = $sample | ConvertTo-Json -Compress

    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payload
        if ($i % 5 -eq 0) { Write-Host "  Enviadas $i predicciones..." }
    } catch {
        Write-Host ("  Error en prediccion $i : " + $_.Exception.Message) -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 100
}
Write-Host "Ventana llena (20 muestras con variacion)" -ForegroundColor Green

# 3. Estado de drift
Write-Header "3. Estado de drift (/drift)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/drift" -Method GET
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 4. Simulacion ANOMALA
Write-Header "4. Simulacion ANOMALA (/simulate/anomaly)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/simulate/anomaly" -Method POST -ContentType "application/json" -Body '{"count":5}'
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 5. Simulacion DRIFT
Write-Header "5. Simulacion DRIFT (/simulate/drift)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/simulate/drift" -Method POST -ContentType "application/json" -Body '{"count":20,"shift_variable":"Alcohol","shift_sigma":3.0}'
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 6. Estado final de drift
Write-Header "6. Estado final de drift (/drift)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/drift" -Method GET
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 7. Resumen monitor
Write-Header "7. Resumen de monitorizacion (/monitor)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/monitor" -Method GET
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 8. Metricas Prometheus
Write-Header "8. Metricas Prometheus (filtradas)"
try {
    $metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET
    ($metrics -split "`n") | Where-Object { $_ -match "drift|anomaly|confidence|entropy|requests" } | Select-Object -First 25
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

# 9. Logs del contenedor
Write-Header "9. Logs del contenedor (ultimas 20 lineas)"
try {
    docker logs vinops_inference_monitored --tail 20
} catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Pruebas Fase 2+3 completadas ===" -ForegroundColor Green
Write-Host "Revisa el grupo de Telegram para ver las alertas enviadas." -ForegroundColor Yellow
