"""
Sistema de alertas para VinOps via Telegram Bot.
Incluye throttling (max 1 alerta por minuto por tipo) para evitar spam.
"""
import os
import time
import requests
from typing import Dict, Any, Optional
from loguru import logger

class AlertManager:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        # Throttling: ultimo envio por tipo de alerta
        self.last_alert_time: Dict[str, float] = {}
        self.throttle_seconds = 60  # 1 minuto entre alertas del mismo tipo

    def _can_send(self, alert_type: str) -> bool:
        """Verifica throttling."""
        now = time.time()
        last = self.last_alert_time.get(alert_type, 0)
        if now - last < self.throttle_seconds:
            return False
        self.last_alert_time[alert_type] = now
        return True

    def send(self, alert_type: str, title: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Envia alerta a Telegram si esta habilitado y pasa throttling."""
        if not self.enabled:
            logger.warning("[ALERTA NO ENVIADA - Bot no configurado] " + title + ": " + message)
            return False

        if not self._can_send(alert_type):
            logger.info("[ALERTA THROTTLED] " + alert_type + ": " + message)
            return False

        # Construir mensaje
        emoji_map = {
            "drift": "🚨",
            "anomaly": "⚠️",
            "operational": "🔧",
            "low_confidence": "📉",
            "simulation": "🧪",
        }
        emoji = emoji_map.get(alert_type, "📢")

        text = emoji + " *" + title + "* " + emoji + "\n\n" + message
        if details:
            text = text + "\n\n*Detalles:*\n"
            for k, v in details.items():
                text = text + "• " + str(k) + ": `" + str(v) + "`\n"

        text = text + "\n⏱ _VinOps Alert System_"

        try:
            resp = requests.post(
                self.base_url + "/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_notification": False,
                },
                timeout=10
            )
            resp.raise_for_status()
            logger.info("[ALERTA ENVIADA] " + alert_type + ": " + title)
            return True
        except Exception as e:
            logger.error("[ALERTA FALLIDA] " + alert_type + ": " + str(e))
            return False

    def send_drift_alert(self, drift_result: Dict[str, Any]):
        """Alerta especifica de drift de datos."""
        status = drift_result.get("status", "UNKNOWN")
        rec = drift_result.get("recommendation", "UNKNOWN")

        if status == "OK":
            return False

        ks = drift_result.get("ks_test", {})
        psi = drift_result.get("psi", {})
        wass = drift_result.get("wasserstein", {})

        title = "Drift de Datos Detectado - " + status
        msg = (
            "Se ha detectado deriva en la distribucion de las muestras de entrada.\n"
            "Recomendacion: *" + rec + "*\n"
            "Ventana evaluada: " + str(drift_result.get("window_size", "?")) + " muestras"
        )
        details = {
            "Variables KS alertadas": ks.get("num_alerted", 0),
            "Variables PSI alertadas": psi.get("num_alerted", 0),
            "Wasserstein distance": wass.get("distance", "N/A"),
            "KS variables": ", ".join(ks.get("variables_alerted", [])) or "Ninguna",
            "PSI variables": ", ".join(psi.get("variables_alerted", [])) or "Ninguna",
        }
        return self.send("drift", title, msg, details)

    def send_anomaly_alert(self, anomaly_count: int, window_size: int):
        """Alerta cuando la tasa de anomalias supera umbral."""
        rate = anomaly_count / window_size if window_size else 0
        title = "Tasa de Anomalias Elevada"
        msg = (
            "La tasa de muestras anomalas en la ventana actual es del *" + str(round(rate*100,1)) + "%* "
            "(" + str(anomaly_count) + "/" + str(window_size) + ")."
        )
        details = {"Tasa": str(round(rate*100,2)) + "%", "Umbral": "25%"}
        return self.send("anomaly", title, msg, details)

    def send_operational_alert(self, metric: str, value: float, threshold: float):
        """Alerta de metrica operativa (CPU/RAM/latencia)."""
        title = "Alerta Operativa: " + metric
        msg = "La metrica *" + metric + "* ha superado el umbral configurado."
        details = {"Valor actual": str(round(value,2)), "Umbral": str(round(threshold,2))}
        return self.send("operational", title, msg, details)

    def send_low_confidence_alert(self, avg_confidence: float, window_size: int):
        """Alerta de baja confianza media en la ventana."""
        title = "Confianza Media Baja"
        msg = (
            "La confianza media del modelo en las ultimas " + str(window_size) + " predicciones "
            "ha caido a *" + str(round(avg_confidence,3)) + "*."
        )
        details = {"Confianza media": str(round(avg_confidence,3)), "Umbral": "0.65"}
        return self.send("low_confidence", title, msg, details)