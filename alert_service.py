"""
Email Alert Service for Air Quality Forecasting app.

Background scheduler that periodically checks all enabled user alerts,
runs predictions for those cities, and sends email notifications when
PM2.5 exceeds the configured threshold.

SMTP configuration via environment variables (or defaults below).
"""

import os
import smtplib
import asyncio
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from collections import defaultdict

from database import get_enabled_alerts_with_emails, mark_alert_sent

logger = logging.getLogger("alert_service")

# ── SMTP settings (override via env vars) ──────────────────────
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "") or SMTP_USER
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"

# How often to run the checker (seconds). Default: every 3 hours.
ALERT_CHECK_INTERVAL = int(os.environ.get("ALERT_CHECK_INTERVAL", "10800"))
# Cooldown per alert – won't re-alert the same (user, city) within this window.
ALERT_COOLDOWN_HOURS = int(os.environ.get("ALERT_COOLDOWN_HOURS", "6"))


def _smtp_configured() -> bool:
    return bool(SMTP_USER and SMTP_PASSWORD)


def _send_email(to_addr: str, subject: str, html_body: str) -> bool:
    """Send a single email via SMTP. Returns True on success."""
    if not _smtp_configured():
        logger.warning("SMTP not configured – skipping email to %s", to_addr)
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_addr], msg.as_string())
        logger.info("Alert email sent to %s", to_addr)
        return True
    except Exception:
        logger.exception("Failed to send email to %s", to_addr)
        return False


def _build_alert_email(username: str, city: str, pm25: float, threshold: float, aqi: str) -> str:
    """Build an HTML email body for a threshold breach."""
    color = {
        "Good": "#4caf50",
        "Moderate": "#ffeb3b",
        "Unhealthy for Sensitive Groups": "#ff9800",
        "Unhealthy": "#f44336",
        "Very Unhealthy": "#9c27b0",
        "Hazardous": "#7e0023",
    }.get(aqi, "#f44336")

    return f"""\
<html>
<body style="font-family:Arial,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:24px;">
  <div style="max-width:520px;margin:auto;background:#16213e;border-radius:12px;padding:28px;">
    <h2 style="margin-top:0;color:#00d4ff;">⚠️ Air Quality Alert</h2>
    <p>Hi <strong>{username}</strong>,</p>
    <p>
      The current predicted PM2.5 for <strong>{city}</strong> is
      <span style="color:{color};font-size:1.3em;font-weight:bold;">{pm25:.1f} µg/m³</span>,
      which exceeds your threshold of <strong>{threshold:.1f} µg/m³</strong>.
    </p>
    <p>AQI Category: <span style="color:{color};font-weight:bold;">{aqi}</span></p>
    <hr style="border-color:#333;">
    <p style="font-size:0.9em;color:#aaa;">
      You received this because you enabled an alert for {city}.<br>
      To stop these emails, disable or remove the alert in your dashboard.
    </p>
  </div>
</body>
</html>"""


def _get_health_category(pm25: float) -> str:
    """Simplified AQI category from PM2.5 value."""
    if pm25 <= 12.0:
        return "Good"
    if pm25 <= 35.4:
        return "Moderate"
    if pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if pm25 <= 150.4:
        return "Unhealthy"
    if pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


async def _run_prediction_for_city(city_name: str) -> float | None:
    """Run the predict endpoint logic for a city and return the city average PM2.5.
    Returns None on failure."""
    try:
        # Import lazily to avoid circular imports at module level
        from app import predict
        # Pass explicit defaults — Query() objects don't resolve when called directly
        response = await predict(city_name=city_name, radius_km=15, hours=336, user=None)
        return response.city_average_pm25
    except Exception:
        logger.exception("Prediction failed for %s", city_name)
        return None


async def check_alerts_once():
    """
    One pass of the alert checker:
      1. Fetch all enabled alerts (respecting cooldown).
      2. Group by city to avoid duplicate predictions.
      3. Run prediction per city.
      4. Compare result against each user's threshold.
      5. Send email if breached; mark cooldown.
    """
    alerts = get_enabled_alerts_with_emails(cooldown_hours=ALERT_COOLDOWN_HOURS)
    if not alerts:
        logger.info("No alerts eligible for checking right now.")
        return

    # Group alerts by city (case-insensitive)
    city_alerts: dict[str, list[dict]] = defaultdict(list)
    for a in alerts:
        city_alerts[a["city_name"].lower()].append(a)

    logger.info("Checking %d alerts across %d cities", len(alerts), len(city_alerts))

    for city_key, alert_list in city_alerts.items():
        city_display = alert_list[0]["city_name"]
        pm25 = await _run_prediction_for_city(city_display)
        if pm25 is None:
            continue

        aqi = _get_health_category(pm25)
        for a in alert_list:
            if pm25 >= a["threshold_pm25"]:
                subject = f"⚠️ Air Quality Alert – {city_display} PM2.5 {pm25:.1f} µg/m³"
                html = _build_alert_email(
                    a["username"], city_display, pm25, a["threshold_pm25"], aqi,
                )
                sent = _send_email(a["email"], subject, html)
                if sent:
                    mark_alert_sent(a["id"])
                    logger.info(
                        "Alerted %s for %s (PM2.5=%.1f > threshold=%.1f)",
                        a["email"], city_display, pm25, a["threshold_pm25"],
                    )
                else:
                    logger.warning(
                        "Email not sent (SMTP disabled?) for %s -> %s",
                        city_display, a["email"],
                    )


async def alert_scheduler():
    """Infinite loop that runs check_alerts_once() every ALERT_CHECK_INTERVAL seconds."""
    if not _smtp_configured():
        logger.warning(
            "SMTP credentials not set (SMTP_USER / SMTP_PASSWORD). "
            "Alert checker will still run but emails won't be sent. "
            "Set env vars SMTP_USER and SMTP_PASSWORD to enable."
        )

    # Small initial delay so the app finishes startup first
    await asyncio.sleep(15)
    logger.info(
        "Alert scheduler started – checking every %d seconds (cooldown %dh)",
        ALERT_CHECK_INTERVAL, ALERT_COOLDOWN_HOURS,
    )

    while True:
        try:
            await check_alerts_once()
        except Exception:
            logger.exception("Alert checker error")
        await asyncio.sleep(ALERT_CHECK_INTERVAL)
