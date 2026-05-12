# ============================================
# test_feedback_loop.ps1
# Banco de pruebas para Fase 5: Feedback Loop
# Ejecutar desde la carpeta observabilidad/
# ============================================

$BaseUrl = "http://localhost:8001"

function Write-Header($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

function Write-Success($msg) { Write-Host "OK: $msg" -ForegroundColor Green }
function Write-Error($msg)   { Write-Host "ERROR: $msg" -ForegroundColor Red }

# ============================================
# PASO 1: Estado inicial
# ============================================
Write-Header "1. Estado inicial (/health)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    $r | ConvertTo-Json -Depth 3
    $initial_count = $r.feedback_summary.count
    Write-Success "Feedback inicial: $initial_count"
} catch {
    Write-Error $_.Exception.Message
    exit 1
}

# ============================================
# PASO 2: Enviar 10 feedbacks del enologo
# ============================================
Write-Header "2. Enviando 10 correcciones de feedback"

# Muestras reales del dataset Wine (clase 1, 2 y 3) con errores simulados
$feedbacks = @(
    @{ features = @{ Alcohol=14.23; MalicAcid=1.71; Ash=2.43; AlcalinityAsh=15.6; Magnesium=127; TotalPhenols=2.80; Flavanoids=3.06; NonflavanoidPhenols=0.28; Proanthocyanins=2.29; ColorIntensity=5.64; Hue=1.04; OD280_OD315=3.92; Proline=1065 }; predicted=1; true=1 },
    @{ features = @{ Alcohol=13.20; MalicAcid=1.78; Ash=2.14; AlcalinityAsh=11.2; Magnesium=100; TotalPhenols=2.65; Flavanoids=2.76; NonflavanoidPhenols=0.26; Proanthocyanins=1.28; ColorIntensity=4.38; Hue=1.05; OD280_OD315=3.40; Proline=1050 }; predicted=1; true=2 },
    @{ features = @{ Alcohol=13.16; MalicAcid=2.36; Ash=2.67; AlcalinityAsh=18.6; Magnesium=101; TotalPhenols=2.80; Flavanoids=3.24; NonflavanoidPhenols=0.30; Proanthocyanins=2.81; ColorIntensity=5.68; Hue=1.03; OD280_OD315=3.17; Proline=1185 }; predicted=2; true=2 },
    @{ features = @{ Alcohol=14.37; MalicAcid=1.95; Ash=2.50; AlcalinityAsh=16.8; Magnesium=113; TotalPhenols=3.85; Flavanoids=3.49; NonflavanoidPhenols=0.24; Proanthocyanins=2.18; ColorIntensity=7.80; Hue=0.86; OD280_OD315=3.45; Proline=1480 }; predicted=1; true=1 },
    @{ features = @{ Alcohol=13.24; MalicAcid=2.59; Ash=2.87; AlcalinityAsh=21.0; Magnesium=118; TotalPhenols=2.80; Flavanoids=2.69; NonflavanoidPhenols=0.39; Proanthocyanins=1.82; ColorIntensity=4.32; Hue=1.04; OD280_OD315=2.93; Proline=735 }; predicted=2; true=3 },
    @{ features = @{ Alcohol=14.20; MalicAcid=1.76; Ash=2.45; AlcalinityAsh=15.2; Magnesium=112; TotalPhenols=3.27; Flavanoids=3.39; NonflavanoidPhenols=0.34; Proanthocyanins=1.97; ColorIntensity=6.75; Hue=1.05; OD280_OD315=2.85; Proline=1450 }; predicted=1; true=1 },
    @{ features = @{ Alcohol=14.39; MalicAcid=1.87; Ash=2.45; AlcalinityAsh=14.6; Magnesium=96; TotalPhenols=2.50; Flavanoids=2.52; NonflavanoidPhenols=0.30; Proanthocyanins=1.98; ColorIntensity=5.25; Hue=1.02; OD280_OD315=3.58; Proline=1290 }; predicted=1; true=2 },
    @{ features = @{ Alcohol=14.06; MalicAcid=2.15; Ash=2.61; AlcalinityAsh=17.6; Magnesium=121; TotalPhenols=2.60; Flavanoids=2.51; NonflavanoidPhenols=0.31; Proanthocyanins=1.25; ColorIntensity=5.05; Hue=1.06; OD280_OD315=3.58; Proline=1295 }; predicted=2; true=2 },
    @{ features = @{ Alcohol=12.85; MalicAcid=1.60; Ash=2.52; AlcalinityAsh=17.7; Magnesium=95; TotalPhenols=2.48; Flavanoids=2.37; NonflavanoidPhenols=0.26; Proanthocyanins=1.46; ColorIntensity=3.93; Hue=1.09; OD280_OD315=3.63; Proline=1015 }; predicted=3; true=3 },
    @{ features = @{ Alcohol=13.50; MalicAcid=1.81; Ash=2.65; AlcalinityAsh=19.0; Magnesium=95; TotalPhenols=2.20; Flavanoids=2.43; NonflavanoidPhenols=0.26; Proanthocyanins=1.57; ColorIntensity=3.80; Hue=1.13; OD280_OD315=3.71; Proline=1010 }; predicted=3; true=2 }
)

for ($i = 0; $i -lt $feedbacks.Count; $i++) {
    $body = @{
        features = $feedbacks[$i].features
        predicted_class = $feedbacks[$i].predicted
        true_class = $feedbacks[$i].true
    } | ConvertTo-Json -Depth 5
    
    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/feedback" -Method POST -ContentType "application/json" -Body $body
        if (($i + 1) % 5 -eq 0) {
            Write-Host "Enviados $($i + 1) feedbacks..."
        }
    } catch {
        Write-Error "Feedback $($i+1): $($_.Exception.Message)"
    }
    Start-Sleep -Milliseconds 100
}
Write-Success "10 feedbacks enviados"

# ============================================
# PASO 3: Verificar acumulacion
# ============================================
Write-Header "3. Estado del feedback (/feedback/status)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/feedback/status" -Method GET
    $r | ConvertTo-Json -Depth 3
    if ($r.count -eq 10) {
        Write-Success "Hay exactamente 10 feedbacks acumulados"
    } else {
        Write-Error "Se esperaban 10, hay $($r.count)"
    }
    if ($r.retrain_recommended) {
        Write-Success "Retraining recomendado: TRUE"
    }
} catch {
    Write-Error $_.Exception.Message
}

# ============================================
# PASO 4: Retraining
# ============================================
Write-Header "4. Retraining del modelo (/retrain)"
Write-Host "Esto puede tardar 10-30 segundos..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/retrain" -Method POST
    $r | ConvertTo-Json -Depth 3
    Write-Success "Retraining completado: $($r.status)"
} catch {
    Write-Error $_.Exception.Message
    Write-Host "Logs del contenedor:" -ForegroundColor Yellow
    docker logs vinops_inference_monitored --tail 20
}

# ============================================
# PASO 5: Recarga del modelo
# ============================================
Write-Header "5. Recarga del modelo (/reload)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/reload" -Method POST
    $r | ConvertTo-Json -Depth 3
    Write-Success "Modelo recargado: $($r.status)"
    Write-Host "Version anterior: $($r.previous_version)"
    Write-Host "Version nueva:    $($r.new_version)"
} catch {
    Write-Error $_.Exception.Message
}

# ============================================
# PASO 6: Verificar nueva version
# ============================================
Write-Header "6. Verificando nueva version (/health)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    $r | ConvertTo-Json -Depth 3
    $new_loaded = $r.model.loaded_at
    Write-Success "Modelo cargado en: $new_loaded"
} catch {
    Write-Error $_.Exception.Message
}

# ============================================
# PASO 7: Prediccion con modelo actualizado
# ============================================
Write-Header "7. Prediccion con modelo actualizado (/predict)"
$payload = @{
    Alcohol = 13.2; MalicAcid = 1.78; Ash = 2.14; AlcalinityAsh = 11.2
    Magnesium = 100; TotalPhenols = 2.65; Flavanoids = 2.76; NonflavanoidPhenols = 0.26
    Proanthocyanins = 1.28; ColorIntensity = 4.38; Hue = 1.05; OD280_OD315 = 3.40
    Proline = 1050
} | ConvertTo-Json -Compress

try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payload
    $r | ConvertTo-Json -Depth 3
    Write-Success "Prediccion: Clase $($r.prediction), confianza $($r.confidence)"
} catch {
    Write-Error $_.Exception.Message
}

Write-Host ""
Write-Host "=== PRUEBAS FEEDBACK LOOP COMPLETADAS ===" -ForegroundColor Green
Write-Host "Revisa Telegram por alertas de 'Retraining Recomendado' y 'Retraining Completado'" -ForegroundColor Cyan