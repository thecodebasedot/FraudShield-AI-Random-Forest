"""Evaluation helpers for FraudShield AI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Score a trained pipeline and return a JSON-serialisable metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "support": {
            "test_size": int(len(y_test)),
            "fraud_in_test": int(y_test.sum()),
        },
        "top_features": _feature_importances(model),
    }


def _feature_importances(model, top_k: int = 10) -> list[tuple[str, float]]:
    """Map the forest's importances back onto readable feature names."""
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocess"]

    # Numeric features pass through; categorical ones are expanded by the OHE.
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], float(importances[i])) for i in order]
