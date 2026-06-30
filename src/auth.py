"""API-key authentication for FraudShield AI.

Keys are random tokens shown to the operator exactly once at creation; only a
SHA-256 hash is stored, so a database leak never exposes usable keys. The
FastAPI dependency ``require_api_key`` guards protected endpoints via the
``X-API-Key`` header.

Create a key from the CLI::

    python -m src.auth create --name acme-bank
"""

from __future__ import annotations

import argparse
import hashlib
import secrets

from sqlalchemy import select

from .db import ApiKey, init_db, session_scope


def hash_key(raw_key: str) -> str:
    """Hash a raw API key for storage / lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_api_key(name: str) -> str:
    """Create a named API key and return the raw token (shown once)."""
    init_db()
    raw_key = "fsk_" + secrets.token_urlsafe(32)
    with session_scope() as session:
        existing = session.scalar(select(ApiKey).where(ApiKey.name == name))
        if existing:
            raise ValueError(f"An API key named {name!r} already exists.")
        session.add(ApiKey(name=name, key_hash=hash_key(raw_key), active=True))
    return raw_key


def verify_api_key(raw_key: str | None) -> str | None:
    """Return the key's name if valid and active, else None."""
    if not raw_key:
        return None
    with session_scope() as session:
        row = session.scalar(
            select(ApiKey).where(
                ApiKey.key_hash == hash_key(raw_key), ApiKey.active.is_(True)
            )
        )
        return row.name if row else None


def list_api_keys() -> list[dict]:
    with session_scope() as session:
        rows = session.scalars(select(ApiKey).order_by(ApiKey.created_at)).all()
        return [
            {"name": r.name, "active": r.active, "created_at": r.created_at.isoformat()}
            for r in rows
        ]


def revoke_api_key(name: str) -> bool:
    with session_scope() as session:
        row = session.scalar(select(ApiKey).where(ApiKey.name == name))
        if not row:
            return False
        row.active = False
        return True


# --------------------------------------------------------------------------- #
# FastAPI dependency
# --------------------------------------------------------------------------- #
def require_api_key():
    """Build the FastAPI dependency.

    Authentication is enforced only if at least one API key exists, so the
    service stays friction-free in local/dev use until you create your first
    key (and immediately locks down afterwards).
    """
    from fastapi import Header, HTTPException

    def _dependency(x_api_key: str | None = Header(default=None)) -> str | None:
        with session_scope() as session:
            from sqlalchemy import func

            from .db import ApiKey as _ApiKey

            key_count = session.scalar(
                select(func.count(_ApiKey.id)).where(_ApiKey.active.is_(True))
            ) or 0

        if key_count == 0:
            return None  # open mode: no keys configured yet

        name = verify_api_key(x_api_key)
        if name is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        return name

    return _dependency


def _main() -> None:
    parser = argparse.ArgumentParser(description="Manage FraudShield API keys")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="create a new API key")
    p_create.add_argument("--name", required=True)

    sub.add_parser("list", help="list API keys")

    p_revoke = sub.add_parser("revoke", help="revoke a key by name")
    p_revoke.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.command == "create":
        raw = create_api_key(args.name)
        print("API key created. Store it now — it will NOT be shown again:\n")
        print(f"  {raw}\n")
    elif args.command == "list":
        for k in list_api_keys():
            status = "active" if k["active"] else "revoked"
            print(f"  {k['name']:<24} {status:<8} {k['created_at']}")
    elif args.command == "revoke":
        ok = revoke_api_key(args.name)
        print("Revoked." if ok else "No such key.")


if __name__ == "__main__":
    _main()
