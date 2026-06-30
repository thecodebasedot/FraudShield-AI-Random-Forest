"""Security & compliance helpers for FraudShield AI (PCI-DSS oriented).

Provides the building blocks a card-data environment needs:

  * **Field encryption** — Fernet (AES-128 in CBC + HMAC) for data at rest.
  * **PII / PAN masking** — never log a full card number; mask to last 4.
  * **Rate limiting** — in-memory token bucket per client, to blunt abuse.
  * **Security headers** — HSTS, no-sniff, frame-deny, etc. for API responses.

See COMPLIANCE.md for how these map to specific PCI-DSS requirements. None of
this turns the project into a certified system — it implements the controls a
real deployment would build on.

Environment
-----------
  FRAUDSHIELD_ENC_KEY   Fernet key for field encryption (generate with the CLI)
  RATE_LIMIT_PER_MINUTE per-client request budget (default 120)
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from threading import Lock

# --------------------------------------------------------------------------- #
# Field encryption (data at rest)
# --------------------------------------------------------------------------- #
def _get_fernet():
    key = os.environ.get("FRAUDSHIELD_ENC_KEY")
    if not key:
        return None
    from cryptography.fernet import Fernet

    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_field(value: str) -> str:
    """Encrypt a string for storage. No-op (returns input) if no key is set."""
    fernet = _get_fernet()
    if fernet is None:
        return value
    return fernet.encrypt(value.encode()).decode()


def decrypt_field(token: str) -> str:
    """Decrypt a value produced by :func:`encrypt_field`."""
    fernet = _get_fernet()
    if fernet is None:
        return token
    return fernet.decrypt(token.encode()).decode()


def generate_key() -> str:
    """Generate a fresh Fernet key (store it in FRAUDSHIELD_ENC_KEY)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


# --------------------------------------------------------------------------- #
# PII / PAN masking
# --------------------------------------------------------------------------- #
_PAN_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def mask_pan(text: str) -> str:
    """Mask anything that looks like a card number (PAN) to its last 4 digits."""

    def _mask(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        if len(digits) < 13:
            return match.group()
        return "*" * (len(digits) - 4) + digits[-4:]

    return _PAN_RE.sub(_mask, text)


def mask_sensitive(record: dict) -> dict:
    """Return a copy of a record safe to log: PANs masked, secrets redacted."""
    safe = {}
    for key, value in record.items():
        if key.lower() in {"password", "secret", "api_key", "x-api-key", "token"}:
            safe[key] = "***REDACTED***"
        elif isinstance(value, str):
            safe[key] = mask_pan(value)
        else:
            safe[key] = value
    return safe


# --------------------------------------------------------------------------- #
# Rate limiting (token bucket per client)
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Simple thread-safe token bucket keyed by client identifier."""

    def __init__(self, per_minute: int | None = None):
        self.capacity = per_minute or int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
        self.refill_per_sec = self.capacity / 60.0
        self._buckets: dict[str, list[float]] = defaultdict(lambda: [self.capacity, time.monotonic()])
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        """Consume a token for ``client_id``; return False if the bucket is empty."""
        with self._lock:
            tokens, last = self._buckets[client_id]
            now = time.monotonic()
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)
            if tokens < 1.0:
                self._buckets[client_id] = [tokens, now]
                return False
            self._buckets[client_id] = [tokens - 1.0, now]
            return True


# --------------------------------------------------------------------------- #
# Security headers
# --------------------------------------------------------------------------- #
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FraudShield security utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("genkey", help="generate a Fernet encryption key")
    args = parser.parse_args()
    if args.command == "genkey":
        print(generate_key())


if __name__ == "__main__":
    _main()
