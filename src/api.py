"""FastAPI REST service for FraudShield AI.

Exposes the trained Random Forest model over HTTP so other systems can score
transactions in real time. Every scored transaction is persisted, API activity
is written to an audit trail, high-risk transactions trigger real-time alerts,
and protected endpoints require an API key once one has been created.

Run it with::

    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import alerts, db
from .auth import require_api_key
from .cache import PredictionCache, cached_score
from .predict import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, FraudDetector

MODEL_PATH = os.environ.get("FRAUDSHIELD_MODEL", DEFAULT_MODEL_PATH)
THRESHOLD = float(os.environ.get("FRAUDSHIELD_THRESHOLD", DEFAULT_THRESHOLD))

app = FastAPI(
    title="FraudShield AI",
    description="Random Forest based fraud detection for bank / online transactions.",
    version="0.2.0",
)

# Loaded lazily on first use so the app can boot even before a model is trained.
_detector: FraudDetector | None = None

# Score cache (Redis if REDIS_URL is set and reachable, else in-memory).
_cache = PredictionCache()

# Shared auth dependency (open until the first API key is created).
api_key_dep = require_api_key()


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def get_detector() -> FraudDetector:
    global _detector
    if _detector is None:
        try:
            _detector = FraudDetector(model_path=MODEL_PATH, threshold=THRESHOLD)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{exc} Train the model first (python -m src.train).",
            ) from exc
    return _detector


def _persist_and_alert(transaction: dict, verdict: dict, api_key_name: str | None) -> None:
    """Record the prediction, audit it and fire alerts — best effort."""
    try:
        db.record_prediction(transaction, verdict, api_key_name=api_key_name)
        db.record_audit(
            action="predict",
            api_key_name=api_key_name,
            detail=f"risk={verdict['risk_level']} p={verdict['fraud_probability']}",
        )
    except Exception:  # persistence must never break scoring
        pass
    alerts.send_alert(transaction, verdict)


class Transaction(BaseModel):
    """One transaction to be scored."""

    amount: float = Field(..., ge=0, examples=[1450.0], description="Transaction amount")
    hour: int = Field(..., ge=0, le=23, examples=[3], description="Hour of day (0-23)")
    txn_count_1h: int = Field(..., ge=0, examples=[7], description="Customer txns in last hour")
    txn_count_24h: int = Field(..., ge=0, examples=[25], description="Customer txns in last 24h")
    foreign_transaction: Literal[0, 1] = Field(..., examples=[1], description="1 if cross-border")
    account_age_days: int = Field(..., ge=0, examples=[12], description="Account age in days")
    is_new_device: Literal[0, 1] = Field(..., examples=[1], description="1 if unseen device")
    merchant_category: str = Field(..., examples=["money_transfer"])
    device_type: str = Field(..., examples=["web"])


class Verdict(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ExplainedVerdict(Verdict):
    explanation: list[FeatureContribution]
    reasons: list[str]


class BatchRequest(BaseModel):
    transactions: list[Transaction]


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service banner."""
    return {
        "service": "FraudShield AI",
        "algorithm": "Random Forest",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Report whether a model is loaded and ready to score."""
    model_ready = os.path.exists(MODEL_PATH)
    return {
        "status": "ok" if model_ready else "model_not_trained",
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "database": db.DATABASE_URL.split("://")[0],
        "cache_backend": _cache.backend,
    }


@app.post("/predict", response_model=Verdict, tags=["scoring"])
def predict(transaction: Transaction, api_key_name: str | None = Depends(api_key_dep)) -> Verdict:
    """Score a single transaction (served from cache when seen recently)."""
    detector = get_detector()
    txn = transaction.model_dump()
    result = cached_score(detector, _cache, txn)
    _persist_and_alert(txn, result, api_key_name)
    return Verdict(**result)


@app.post("/predict/batch", response_model=list[Verdict], tags=["scoring"])
def predict_batch(
    request: BatchRequest, api_key_name: str | None = Depends(api_key_dep)
) -> list[Verdict]:
    """Score many transactions in one call."""
    detector = get_detector()
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    txns = [t.model_dump() for t in request.transactions]
    results = detector.score_many(txns)
    for txn, result in zip(txns, results):
        _persist_and_alert(txn, result, api_key_name)
    return [Verdict(**r) for r in results]


@app.post("/explain", response_model=ExplainedVerdict, tags=["scoring"])
def explain(
    transaction: Transaction, api_key_name: str | None = Depends(api_key_dep)
) -> ExplainedVerdict:
    """Score a transaction and explain it with SHAP feature attributions."""
    # Imported here so the heavier SHAP dependency only loads if /explain is used.
    from .explain import get_explainer

    # Reuse the same model the detector validated is present.
    get_detector()
    txn = transaction.model_dump()
    explainer = get_explainer(MODEL_PATH)
    result = explainer.explain(txn)
    _persist_and_alert(txn, result, api_key_name)
    return ExplainedVerdict(**result)


@app.get("/stats", tags=["admin"])
def stats(api_key_name: str | None = Depends(api_key_dep)) -> dict:
    """Aggregate analytics over everything scored so far."""
    return db.stats_summary()


@app.get("/predictions", tags=["admin"])
def predictions(
    limit: int = 50, api_key_name: str | None = Depends(api_key_dep)
) -> list[dict]:
    """Most recent scored transactions (newest first)."""
    limit = max(1, min(limit, 500))
    return db.recent_predictions(limit=limit)
