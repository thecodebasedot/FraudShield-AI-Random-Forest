"""FastAPI REST service for FraudShield AI.

Exposes the trained Random Forest model over HTTP so other systems can score
transactions in real time.

Run it with::

    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import CATEGORICAL_FEATURES
from .predict import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, FraudDetector

MODEL_PATH = os.environ.get("FRAUDSHIELD_MODEL", DEFAULT_MODEL_PATH)
THRESHOLD = float(os.environ.get("FRAUDSHIELD_THRESHOLD", DEFAULT_THRESHOLD))

app = FastAPI(
    title="FraudShield AI",
    description="Random Forest based fraud detection for bank / online transactions.",
    version="0.1.0",
)

# Loaded lazily on first use so the app can boot even before a model is trained.
_detector: FraudDetector | None = None


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
    }


@app.post("/predict", response_model=Verdict, tags=["scoring"])
def predict(transaction: Transaction) -> Verdict:
    """Score a single transaction."""
    detector = get_detector()
    result = detector.score(transaction.model_dump())
    return Verdict(**result)


@app.post("/predict/batch", response_model=list[Verdict], tags=["scoring"])
def predict_batch(request: BatchRequest) -> list[Verdict]:
    """Score many transactions in one call."""
    detector = get_detector()
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    results = detector.score_many([t.model_dump() for t in request.transactions])
    return [Verdict(**r) for r in results]


@app.post("/explain", response_model=ExplainedVerdict, tags=["scoring"])
def explain(transaction: Transaction) -> ExplainedVerdict:
    """Score a transaction and explain it with SHAP feature attributions."""
    # Imported here so the heavier SHAP dependency only loads if /explain is used.
    from .explain import get_explainer

    # Reuse the same model the detector validated is present.
    get_detector()
    explainer = get_explainer(MODEL_PATH)
    result = explainer.explain(transaction.model_dump())
    return ExplainedVerdict(**result)
