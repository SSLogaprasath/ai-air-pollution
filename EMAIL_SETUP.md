# Email Alert Service – Setup Guide

This application includes a background alert service that monitors air quality predictions and sends email notifications when PM2.5 levels exceed user-configured thresholds.

## Prerequisites

- A working SMTP email account (Gmail recommended, but any SMTP provider works)
- Python environment with the app dependencies installed

## Configuration

All email settings are configured via **environment variables**. Create a `.env` file in the project root or export them in your shell.

### Required Variables

| Variable        | Description                          | Example                        |
|-----------------|--------------------------------------|--------------------------------|
| `SMTP_USER`     | SMTP login username / email address  | `yourname@gmail.com`           |
| `SMTP_PASSWORD`  | SMTP login password or app password  | `abcd efgh ijkl mnop`         |

### Optional Variables

| Variable              | Description                                      | Default            |
|-----------------------|--------------------------------------------------|--------------------|
| `SMTP_HOST`           | SMTP server hostname                             | `smtp.gmail.com`   |
| `SMTP_PORT`           | SMTP server port                                 | `587`              |
| `SMTP_FROM`           | "From" address on outgoing emails                | Same as `SMTP_USER`|
| `SMTP_USE_TLS`        | Enable STARTTLS (`true` / `false`)               | `true`             |
| `ALERT_CHECK_INTERVAL`| Seconds between alert checks                     | `10800` (3 hours)  |
| `ALERT_COOLDOWN_HOURS`| Hours before re-alerting the same user+city pair | `6`                |

### Example `.env` file

```
SMTP_USER=yourname@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=yourname@gmail.com
SMTP_USE_TLS=true
ALERT_CHECK_INTERVAL=10800
ALERT_COOLDOWN_HOURS=6
```

> **Note:** The `.env` file is listed in `.gitignore` and will not be committed.

## Gmail Setup (Recommended)

If you use Gmail as your SMTP provider:

1. **Enable 2-Step Verification** on your Google account:
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Under "Signing in to Google", enable **2-Step Verification**

2. **Generate an App Password:**
   - Go to [App Passwords](https://myaccount.google.com/apppasswords)
   - Select **Mail** as the app and your device type
   - Click **Generate**
   - Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

3. **Set the environment variables:**
   ```
   SMTP_USER=yourname@gmail.com
   SMTP_PASSWORD=abcd efgh ijkl mnop
   ```

> **Important:** Do NOT use your regular Gmail password. Always use an App Password.

## Other SMTP Providers

### Outlook / Office 365

```
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USER=yourname@outlook.com
SMTP_PASSWORD=your_password
```

### Custom / Self-hosted SMTP

```
SMTP_HOST=mail.yourdomain.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your_password
```

## How It Works

1. The alert scheduler starts automatically as a background task when the app launches.
2. Every `ALERT_CHECK_INTERVAL` seconds (default: 3 hours), it:
   - Fetches all enabled user alerts from the database
   - Skips alerts that were already sent within the cooldown window
   - Runs air quality predictions for each city
   - Compares predicted PM2.5 against each user's threshold
   - Sends an HTML email notification if the threshold is exceeded
3. After sending, the alert is marked with a cooldown to prevent duplicate emails.

## Verifying the Setup

1. Start the app:
   ```
   python app.py
   ```
2. Check the console logs for:
   ```
   Alert scheduler started – checking every 10800 seconds (cooldown 6h)
   ```
   If SMTP is not configured, you'll see:
   ```
   SMTP credentials not set (SMTP_USER / SMTP_PASSWORD). Alert checker will still run but emails won't be sent.
   ```
3. Create an alert via the dashboard with a low PM2.5 threshold to trigger a test email.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "SMTP not configured" in logs | Set `SMTP_USER` and `SMTP_PASSWORD` environment variables |
| Gmail authentication error | Use an App Password, not your regular password |
| "Connection refused" | Check `SMTP_HOST` and `SMTP_PORT` values |
| Emails not arriving | Check spam/junk folder; verify the "From" address |
| Alerts not triggering | Ensure alerts are enabled in the dashboard and cooldown has elapsed |
