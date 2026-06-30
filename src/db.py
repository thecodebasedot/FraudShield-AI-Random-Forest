"""Persistence layer for FraudShield AI.

Stores every scored transaction, an audit trail of API activity and the API
keys used to authenticate. Backed by SQLAlchemy so the same code runs on
SQLite (default, zero-config) or PostgreSQL in production — just point
``DATABASE_URL`` at it::

    export DATABASE_URL=postgresql+psycopg://user:pass@host/fraudshield

Defaults to ``sqlite:///fraudshield.db`` in the project root.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///fraudshield.db")

# check_same_thread is a SQLite-only knob; harmless to pass only for sqlite.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One scored transaction and its verdict."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    # Transaction features (denormalized for easy querying / analytics).
    amount: Mapped[float] = mapped_column(Float)
    hour: Mapped[int] = mapped_column(Integer)
    txn_count_1h: Mapped[int] = mapped_column(Integer)
    txn_count_24h: Mapped[int] = mapped_column(Integer)
    foreign_transaction: Mapped[int] = mapped_column(Integer)
    account_age_days: Mapped[int] = mapped_column(Integer)
    is_new_device: Mapped[int] = mapped_column(Integer)
    merchant_category: Mapped[str] = mapped_column(String(64))
    device_type: Mapped[str] = mapped_column(String(32))

    # Verdict.
    fraud_probability: Mapped[float] = mapped_column(Float, index=True)
    is_fraud: Mapped[bool] = mapped_column(Boolean, index=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)

    api_key_name: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditLog(Base):
    """An audit-trail entry for anything noteworthy that happens via the API."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    api_key_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(String(512), nullable=True)


class ApiKey(Base):
    """A hashed API key. The raw key is shown once at creation, never stored."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


def init_db() -> None:
    """Create tables if they don't exist yet."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Write helpers
# --------------------------------------------------------------------------- #
def record_prediction(transaction: dict, verdict: dict, api_key_name: str | None = None) -> int:
    """Persist a scored transaction; returns the new row id."""
    with session_scope() as session:
        row = Prediction(
            amount=transaction["amount"],
            hour=transaction["hour"],
            txn_count_1h=transaction["txn_count_1h"],
            txn_count_24h=transaction["txn_count_24h"],
            foreign_transaction=transaction["foreign_transaction"],
            account_age_days=transaction["account_age_days"],
            is_new_device=transaction["is_new_device"],
            merchant_category=transaction["merchant_category"],
            device_type=transaction["device_type"],
            fraud_probability=verdict["fraud_probability"],
            is_fraud=verdict["is_fraud"],
            risk_level=verdict["risk_level"],
            api_key_name=api_key_name,
        )
        session.add(row)
        session.flush()
        return row.id


def record_audit(action: str, api_key_name: str | None = None, detail: str | None = None) -> None:
    """Append an audit-trail entry."""
    with session_scope() as session:
        session.add(AuditLog(action=action, api_key_name=api_key_name, detail=detail))


# --------------------------------------------------------------------------- #
# Read helpers (analytics)
# --------------------------------------------------------------------------- #
def stats_summary() -> dict:
    """Aggregate stats for the admin dashboard."""
    with session_scope() as session:
        total = session.scalar(select(func.count(Prediction.id))) or 0
        fraud = session.scalar(
            select(func.count(Prediction.id)).where(Prediction.is_fraud.is_(True))
        ) or 0
        avg_prob = session.scalar(select(func.avg(Prediction.fraud_probability)))

        by_risk = dict(
            session.execute(
                select(Prediction.risk_level, func.count(Prediction.id)).group_by(
                    Prediction.risk_level
                )
            ).all()
        )
        return {
            "total_transactions": int(total),
            "fraud_flagged": int(fraud),
            "fraud_rate": (float(fraud) / total) if total else 0.0,
            "avg_fraud_probability": float(avg_prob) if avg_prob is not None else 0.0,
            "by_risk_level": {k: int(v) for k, v in by_risk.items()},
        }


def recent_predictions(limit: int = 50) -> list[dict]:
    """Most recent predictions, newest first."""
    with session_scope() as session:
        rows = session.scalars(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "amount": r.amount,
                "merchant_category": r.merchant_category,
                "device_type": r.device_type,
                "fraud_probability": r.fraud_probability,
                "is_fraud": r.is_fraud,
                "risk_level": r.risk_level,
            }
            for r in rows
        ]
