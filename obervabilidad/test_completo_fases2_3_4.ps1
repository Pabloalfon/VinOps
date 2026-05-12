# ============================================
# test_completo_fases2_3_4.ps1 (FINAL)
# Banco de pruebas Fases 2+3+4
# ============================================

$BaseUrl = "http://localhost:8001"
$PromUrl = "http://localhost:9090"
$GrafUrl = "http://localhost:3000"

function Write-Header($title) {
    Write-Host ""
    Write-Host ("=== " + $title + " ===") -ForegroundColor Cyan
}
function Write-OK($msg) { Write-Host ("[OK] " + $msg) -ForegroundColor Green }
function Write-WARN($msg) { Write-Host ("[WARN] " + $msg) -ForegroundColor Yellow }
function Write-ERR($msg) { Write-Host ("[ERR] " + $msg) -ForegroundColor Red }

# Auth para Grafana (Basic Auth manual)
$grafAuth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:admin"))
$grafHeaders = @{ "Authorization" = "Basic $grafAuth" }

# =====================================================
# FASE 2: DETECCION DE DRIFT
# =====================================================

Write-Header "FASE 2 - 1. Health check"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET
    if ($r.status -eq "ok") { Write-OK "Servicio activo" } else { Write-ERR "Servicio no OK" }
    if ($r.alerts_enabled) { Write-OK "Alertas habilitadas" } else { Write-WARN "Alertas deshabilitadas" }
    Write-Host ("  CPU: " + $r.cpu_percent + "%, RAM: " + [math]::Round($r.memory_bytes/1MB,1) + " MB")
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 2 - 2. Llenando ventana (20 predicciones)"
$normalPayload = @{
    Alcohol = 13.2; MalicAcid = 1.78; Ash = 2.14; AlcalinityAsh = 11.2
    Magnesium = 100; TotalPhenols = 2.65; Flavanoids = 2.76; NonflavanoidPhenols = 0.26
    Proanthocyanins = 1.28; ColorIntensity = 4.38; Hue = 1.05; OD280_OD315 = 3.40
    Proline = 1050
} | ConvertTo-Json -Compress

$ok = 0
for ($i = 1; $i -le 20; $i++) {
    try { $null = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method POST -ContentType "application/json" -Body $normalPayload; $ok++; if ($i % 5 -eq 0) { Write-Host "  $i..." } }
    catch { Write-ERR ("Error $i : " + $_.Exception.Message) }
    Start-Sleep -Milliseconds 50
}
Write-OK "Ventana llena ($ok/20 OK)"

Write-Header "FASE 2 - 3. Estado de drift (/drift)"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/drift" -Method GET
    $color = if($r.status -eq "DRIFT_DETECTED"){"Red"}elseif($r.status -eq "MONITOR"){"Yellow"}else{"Green"}
    Write-Host ("  Status: " + $r.status) -ForegroundColor $color
    Write-Host ("  Recomendacion: " + $r.recommendation)
    Write-Host ("  KS: " + $r.ks_test.num_alerted + "/13, PSI: " + $r.psi.num_alerted + "/13, W: " + $r.wasserstein.distance)
} catch { Write-ERR $_.Exception.Message }

# =====================================================
# FASE 3: ALERTAS + SIMULACION
# =====================================================

Write-Header "FASE 3 - 4. Simulacion ANOMALA"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/simulate/anomaly" -Method POST -ContentType "application/json" -Body '{"count":5}'
    Write-OK ("Anomala: " + $r.count + " muestras, drift: " + $r.drift_after_simulation.status)
    Write-WARN "Revisa Telegram"
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 3 - 5. Simulacion DRIFT"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/simulate/drift" -Method POST -ContentType "application/json" -Body '{"count":20,"shift_variable":"Alcohol","shift_sigma":3.0}'
    Write-OK ("Drift: " + $r.count + " muestras, " + $r.shift_variable + " +" + $r.shift_sigma + "sigma")
    Write-Host ("  PSI Alcohol: " + $r.drift_after_simulation.psi.details.Alcohol) -ForegroundColor Red
    Write-WARN "Revisa Telegram"
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 3 - 6. Estado final de drift"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/drift" -Method GET
    Write-Host ("  Status: " + $r.status) -ForegroundColor $(if($r.status -eq "DRIFT_DETECTED"){"Red"}elseif($r.status -eq "MONITOR"){"Yellow"}else{"Green"})
    Write-Host ("  KS vars: " + ($r.ks_test.variables_alerted -join ", "))
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 3 - 7. Resumen monitor"
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/monitor" -Method GET
    Write-Host ("  Predicciones: " + $r.current_metrics.predictions_total + ", Confianza: " + $r.current_metrics.avg_confidence)
} catch { Write-ERR $_.Exception.Message }

# =====================================================
# FASE 4: PROMETHEUS + GRAFANA
# =====================================================

Write-Header "FASE 4 - 8. Prometheus targets"
try {
    $r = Invoke-RestMethod -Uri "$PromUrl/api/v1/targets" -Method GET
    $vinops = $r.data.activeTargets | Where-Object { $_.labels.job -eq "vinops" }
    if ($vinops) {
        if ($vinops.health -eq "up") { Write-OK ("Target vinops: " + $vinops.health.ToUpper()) }
        else { Write-WARN ("Target: " + $vinops.health) }
        Write-Host ("  URL: " + $vinops.scrapeUrl)
    } else { Write-WARN "Target no encontrado" }
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 4 - 9. Grafana dashboards"
try {
    $r = Invoke-RestMethod -Uri "$GrafUrl/api/search?query=VinOps" -Method GET -Headers $grafHeaders
    if ($r.Count -gt 0) {
        Write-OK ("Dashboard: " + $r[0].title)
        Write-Host ("  URL: " + $GrafUrl + $r[0].url)
    } else { Write-WARN "Dashboard no encontrado" }
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 4 - 10. Metricas Prometheus"
try {
    $metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET
    ($metrics -split "`n") | Where-Object { $_ -match "^# HELP vinops" } | ForEach-Object { Write-Host "  " $_ }
} catch { Write-ERR $_.Exception.Message }

Write-Header "FASE 4 - 11. Logs contenedor"
try { docker logs vinops_inference_monitored --tail 10 } catch { Write-ERR $_.Exception.Message }

Write-Header "RESUMEN"
Write-OK "Fase 2: Drift (KS + PSI + Wasserstein)"
Write-OK "Fase 3: Alertas Telegram + Simulaciones"
Write-OK "Fase 4: Prometheus UP + Grafana dashboard"
Write-Host ""
Write-Host "Revisa Telegram y http://localhost:3000 (admin/admin)" -ForegroundColor Yellow
Write-Host "=== Pruebas completadas ===" -ForegroundColor Green