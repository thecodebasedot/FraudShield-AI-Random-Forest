"""Score new transactions with a trained FraudShield AI model.

Loads the persisted pipeline and exposes a small API plus a CLI so a single
transaction (described as JSON) can be flagged as fraud or not.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import pandas as pd

from .model import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = "models/fraudshield_rf.joblib"

# Probability at/above which a transaction is flagged. Tunable per risk appetite.
DEFAULT_THRESHOLD = 0.5


class FraudDetector:
    """Thin wrapper around a persisted pipeline for scoring transactions."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = DEFAULT_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model at {model_path!r}. Train one first: python -m src.train"
            )
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def score(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Score one transaction dict and return the fraud decision."""
        return self.score_many([transaction])[0]

    def score_many(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a batch of transaction dicts."""
        frame = pd.DataFrame(transactions)
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Transaction is missing required fields: {missing}")

        proba = self.model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        results = []
        for p in proba:
            results.append(
                {
                    "fraud_probability": round(float(p), 4),
                    "is_fraud": bool(p >= self.threshold),
                    "risk_level": _risk_level(float(p)),
                }
            )
        return results


def _risk_level(p: float) -> str:
    if p >= 0.8:
        return "HIGH"
    if p >= 0.5:
        return "MEDIUM"
    if p >= 0.2:
        return "LOW"
    return "MINIMAL"


def _example_transaction() -> dict[str, Any]:
    """A representative transaction used when the CLI is run without input."""
    return {
        "amount": 1450.00,
        "hour": 3,
        "txn_count_1h": 7,
        "txn_count_24h": 25,
        "foreign_transaction": 1,
        "account_age_days": 12,
        "is_new_device": 1,
        "merchant_category": "money_transfer",
        "device_type": "web",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a transaction for fraud")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--json",
        default=None,
        help="transaction as a JSON string; uses a built-in example if omitted",
    )
    args = parser.parse_args()

    transaction = json.loads(args.json) if args.json else _example_transaction()
    detector = FraudDetector(model_path=args.model, threshold=args.threshold)
    result = detector.score(transaction)

    print("Transaction:")
    print(json.dumps(transaction, indent=2))
    print("\nFraudShield AI verdict:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
