
# test_shadow.ps1
# Banco de pruebas para Bonus: Shadow Testing (Champion-Challenger)
# Ejecutar desde bonus_shadowTesting/ con el contenedor levantado

$BaseUrl = "http://localhost:8002"

function Write-Header($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

function Write-Success($msg) { Write-Host "OK: $msg" -ForegroundColor Green }
function Write-Error($msg)   { Write-Host "ERROR: $msg" -ForegroundColor Red }


# PASO 1: Estado inicial
Write-Header "1. Estado inicial (/health)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    $r | ConvertTo-Json -Depth 3
    if ($r.champion_loaded -and $r.challenger_loaded) {
        Write-Success "Champion y Challenger cargados. Shadow habilitado."
    } else {
        Write-Error "Falta Champion o Challenger."
        exit 1
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
}


# PASO 2: Prediccion con shadow (Champion responde, Challenger predice)
Write-Header "2. Prediccion con shadow (/predict)"
$payload = @{
    Alcohol = 13.2; MalicAcid = 1.78; Ash = 2.14; AlcalinityAsh = 11.2
    Magnesium = 100; TotalPhenols = 2.65; Flavanoids = 2.76; NonflavanoidPhenols = 0.26
    Proanthocyanins = 1.28; ColorIntensity = 4.38; Hue = 1.05; OD280_OD315 = 3.40
    Proline = 1050
} | ConvertTo-Json -Compress

try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payload
    $r | ConvertTo-Json -Depth 3
    if ($r.model -eq "champion") {
        Write-Success "Champion respondio al cliente."
    }
    if ($r.shadow_comparison) {
        Write-Success "Challenger predijo en shadow: clase $($r.shadow_comparison.challenger_prediction)"
        Write-Success "Acuerdo: $($r.shadow_comparison.agreement), Delta confianza: $($r.shadow_comparison.confidence_delta)"
    }
} catch {
    Write-Error $_.Exception.Message
}


# PASO 3: Enviar 10 predicciones para acumular comparaciones
Write-Header "3. Acumulando 10 comparaciones"
for ($i = 1; $i -le 10; $i++) {
    # Variar ligeramente Alcohol para simular diferentes muestras
    $variant = @{
        Alcohol = [math]::Round(12.5 + (Get-Random -Maximum 3.0), 2)
        MalicAcid = 1.78; Ash = 2.14; AlcalinityAsh = 11.2
        Magnesium = 100; TotalPhenols = 2.65; Flavanoids = 2.76
        NonflavanoidPhenols = 0.26; Proanthocyanins = 1.28
        ColorIntensity = 4.38; Hue = 1.05; OD280_OD315 = 3.40
        Proline = 1050
    } | ConvertTo-Json -Compress
    
    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $variant
        if ($i % 5 -eq 0) { Write-Host "Enviadas $i predicciones..." }
    } catch {
        Write-Error "Prediccion $i fallo: $($_.Exception.Message)"
    }
    Start-Sleep -Milliseconds 50
}
Write-Success "10 predicciones enviadas"


# PASO 4: Ver estadisticas acumuladas
Write-Header "4. Estadisticas shadow (/shadow/status)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/shadow/status" -Method GET
    $r | ConvertTo-Json -Depth 3
    $stats = $r.statistics
    Write-Success "Comparaciones: $($stats.comparisons)"
    Write-Success "Tasa de acuerdo: $($stats.agreement_rate) ($($stats.agreements)/$($stats.disagreements))"
    Write-Success "Confianza Champion: $($stats.avg_confidence_champion)"
    Write-Success "Confianza Challenger: $($stats.avg_confidence_challenger)"
    Write-Success "Latencia Champion: $($stats.avg_latency_champion_ms) ms"
    Write-Success "Latencia Challenger: $($stats.avg_latency_challenger_ms) ms"
    if ($stats.challenger_better_confidence -gt $stats.champion_better_confidence) {
        Write-Host "Challenger tiene mejor confianza en $($stats.challenger_better_confidence) casos" -ForegroundColor Green
    }
} catch {
    Write-Error $_.Exception.Message
}


# PASO 5: Promover Challenger a Champion
Write-Header "5. Promoviendo Challenger (/shadow/promote)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/shadow/promote" -Method POST
    $r | ConvertTo-Json -Depth 3
    Write-Success "Challenger promovido a Champion."
} catch {
    Write-Error $_.Exception.Message
}


# PASO 6: Verificar nuevo Champion
Write-Header "6. Verificando nuevo Champion (/health)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    $r | ConvertTo-Json -Depth 3
    Write-Success "Nuevo Champion activo. Shadow sigue habilitado."
} catch {
    Write-Error $_.Exception.Message
}


# PASO 7: Prediccion con nuevo Champion
Write-Header "7. Prediccion con nuevo Champion (/predict)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $payload
    $r | ConvertTo-Json -Depth 3
    Write-Success "Nuevo Champion predice correctamente."
} catch {
    Write-Error $_.Exception.Message
}

Write-Host ""
Write-Host "=== PRUEBAS SHADOW TESTING COMPLETADAS ===" -ForegroundColor Green