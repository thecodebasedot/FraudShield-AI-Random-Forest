"""Real-time alerting for FraudShield AI.

When a transaction is flagged as high-risk, notify a human channel. Supports
Slack (incoming webhook) and email (SMTP); both are configured via environment
variables and are entirely optional. If nothing is configured, alerts fall back
to the application log, so the call is always safe and never blocks scoring.

Environment variables
---------------------
  FRAUDSHIELD_ALERT_LEVEL   minimum risk level to alert on (default: HIGH)
  SLACK_WEBHOOK_URL         Slack incoming-webhook URL
  SMTP_HOST, SMTP_PORT      SMTP server (port default 587)
  SMTP_USER, SMTP_PASSWORD  SMTP credentials
  ALERT_EMAIL_FROM          sender address
  ALERT_EMAIL_TO            comma-separated recipient list
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

logger = logging.getLogger("fraudshield.alerts")

# Ordering so we can compare "is this at least as severe as the threshold".
_RISK_ORDER = {"MINIMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _should_alert(risk_level: str) -> bool:
    threshold = os.environ.get("FRAUDSHIELD_ALERT_LEVEL", "HIGH").upper()
    return _RISK_ORDER.get(risk_level, 0) >= _RISK_ORDER.get(threshold, 3)


def _format_message(transaction: dict, verdict: dict) -> str:
    prob = verdict["fraud_probability"]
    return (
        f"🚨 FraudShield AI alert — {verdict['risk_level']} risk "
        f"({prob:.0%} fraud probability)\n"
        f"Amount: {transaction.get('amount')} | "
        f"Hour: {transaction.get('hour')} | "
        f"Merchant: {transaction.get('merchant_category')} | "
        f"Device: {transaction.get('device_type')} | "
        f"Foreign: {transaction.get('foreign_transaction')} | "
        f"New device: {transaction.get('is_new_device')}"
    )


def send_alert(transaction: dict, verdict: dict) -> bool:
    """Send an alert if the verdict is severe enough.

    Returns True if at least one channel (or the log fallback) handled it.
    Never raises — alerting must not break the scoring path.
    """
    if not _should_alert(verdict.get("risk_level", "MINIMAL")):
        return False

    message = _format_message(transaction, verdict)
    delivered = False

    try:
        if os.environ.get("SLACK_WEBHOOK_URL"):
            _send_slack(message)
            delivered = True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Slack alert failed: %s", exc)

    try:
        if os.environ.get("SMTP_HOST") and os.environ.get("ALERT_EMAIL_TO"):
            _send_email("FraudShield AI: high-risk transaction", message)
            delivered = True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Email alert failed: %s", exc)

    if not delivered:
        # Fallback so the alert is never silently lost.
        logger.warning(message)
    return True


def _send_slack(message: str) -> None:
    url = os.environ["SLACK_WEBHOOK_URL"]
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)  # noqa: S310 - trusted, operator-set URL


def _send_email(subject: str, body: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("ALERT_EMAIL_FROM", user or "fraudshield@localhost")
    recipients = [a.strip() for a in os.environ["ALERT_EMAIL_TO"].split(",") if a.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)
