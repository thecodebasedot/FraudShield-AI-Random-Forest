"""Per-transaction explainability for FraudShield AI using SHAP.

A fraud score on its own is hard to act on — analysts need to know *why* a
transaction was flagged. This module uses SHAP TreeExplainer on the Random
Forest to attribute the prediction to individual features, so each verdict
comes with a human-readable "reasons" list.

Usage::

    python -m src.explain                      # explain a built-in risky example
    python -m src.explain --json '{...}'       # explain a custom transaction
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .model import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .predict import DEFAULT_MODEL_PATH, FraudDetector


class TransactionExplainer:
    """Wraps a trained pipeline with a SHAP explainer over its forest."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.detector = FraudDetector(model_path=model_path)
        self.model = self.detector.model
        self.preprocessor = self.model.named_steps["preprocess"]
        self.classifier = self.model.named_steps["classifier"]
        self.feature_names = self._expanded_feature_names()
        self._explainer = None  # built lazily; importing shap is a little heavy

    def _expanded_feature_names(self) -> list[str]:
        ohe = self.preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        return NUMERIC_FEATURES + cat_names

    @property
    def explainer(self):
        if self._explainer is None:
            import shap  # imported lazily to keep base predict path lightweight

            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def explain(self, transaction: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        """Return the verdict plus the top features pushing it toward fraud."""
        frame = pd.DataFrame([transaction])
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Transaction is missing required fields: {missing}")

        verdict = self.detector.score(transaction)

        transformed = self.preprocessor.transform(frame[FEATURE_COLUMNS])
        transformed = _to_dense(transformed)

        # SHAP values for the positive (fraud) class.
        shap_values = self._fraud_class_shap(transformed)[0]

        contributions = sorted(
            zip(self.feature_names, shap_values),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_k]

        verdict["explanation"] = [
            {
                "feature": name,
                "shap_value": round(float(value), 4),
                "direction": "increases_risk" if value > 0 else "decreases_risk",
            }
            for name, value in contributions
        ]
        verdict["reasons"] = [
            _humanize(name, value, transaction)
            for name, value in contributions
            if value > 0
        ]
        return verdict

    def _fraud_class_shap(self, transformed: np.ndarray) -> np.ndarray:
        """Normalize SHAP output shape across shap/sklearn versions."""
        values = self.explainer.shap_values(transformed)
        # Newer shap returns a single (n, features, classes) array; older returns
        # a list per class. Reduce both to the fraud-class (index 1) matrix.
        if isinstance(values, list):
            return np.asarray(values[1])
        values = np.asarray(values)
        if values.ndim == 3:
            return values[:, :, 1]
        return values


def _to_dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _humanize(feature: str, value: float, txn: dict[str, Any]) -> str:
    """Turn a SHAP attribution into a short analyst-friendly reason."""
    readable = {
        "amount": "high transaction amount",
        "hour": "unusual transaction hour",
        "txn_count_1h": "high transaction velocity (1h)",
        "txn_count_24h": "high transaction velocity (24h)",
        "foreign_transaction": "foreign / cross-border transaction",
        "account_age_days": "young account",
        "is_new_device": "new / unseen device",
    }
    if feature in readable:
        return readable[feature]
    if feature.startswith("merchant_category_"):
        return f"merchant category '{feature.split('_', 2)[-1]}'"
    if feature.startswith("device_type_"):
        return f"device type '{feature.split('_', 2)[-1]}'"
    return feature


def _example_transaction() -> dict[str, Any]:
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


@lru_cache(maxsize=1)
def get_explainer(model_path: str = DEFAULT_MODEL_PATH) -> TransactionExplainer:
    """Cached explainer so repeated calls don't rebuild the SHAP tree."""
    return TransactionExplainer(model_path=model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a fraud prediction")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--json", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model!r}. Train one: python -m src.train")

    transaction = json.loads(args.json) if args.json else _example_transaction()
    explainer = TransactionExplainer(model_path=args.model)
    result = explainer.explain(transaction, top_k=args.top_k)

    print("Transaction:")
    print(json.dumps(transaction, indent=2))
    print("\nVerdict + explanation:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
