"""Quick test of alert service components."""
from database import get_enabled_alerts_with_emails
from alert_service import _build_alert_email, _get_health_category, _smtp_configured

# Test 1: DB query
alerts = get_enabled_alerts_with_emails(cooldown_hours=6)
print(f"Eligible alerts: {len(alerts)}")
for a in alerts:
    print(f"  Alert id={a['id']}: user={a['username']} ({a['email']}), "
          f"city={a['city_name']}, threshold={a['threshold_pm25']}")

# Test 2: Email template
pm25 = 85.0
aqi = _get_health_category(pm25)
html = _build_alert_email("testuser", "Delhi", pm25, 55.4, aqi)
print(f"\nSample email: PM2.5={pm25}, AQI={aqi}")
print(f"HTML length: {len(html)} chars")
print(f"Contains threshold text: {'exceeds your threshold' in html}")

# Test 3: SMTP status
print(f"\nSMTP configured: {_smtp_configured()}")
print("(Set SMTP_USER and SMTP_PASSWORD env vars to enable email sending)")
