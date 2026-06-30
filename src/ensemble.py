"""Ensemble model for FraudShield AI: Random Forest + XGBoost + Gradient Boosting.

A soft-voting ensemble averages the probabilities of three complementary tree
models. XGBoost in particular tends to squeeze extra recall out of the rare
fraud class, and blending it with the Random Forest usually beats either alone.

The ensemble is saved separately (``models/fraudshield_ensemble.joblib``) and is
a drop-in for ``FraudDetector`` — it only needs ``predict_proba``.

Usage::

    python -m src.ensemble                      # train + evaluate (synthetic data)
    python -m src.ensemble --data data/transactions.csv --save
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .data_generator import generate_transactions
from .evaluate import evaluate
from .model import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

DEFAULT_ENSEMBLE_PATH = "models/fraudshield_ensemble.joblib"


def build_ensemble(
    scale_pos_weight: float = 10.0,
    rf_estimators: int = 200,
    xgb_estimators: int = 300,
    random_state: int = 42,
):
    """Build the preprocessing + soft-voting ensemble pipeline.

    ``scale_pos_weight`` is XGBoost's imbalance knob (roughly n_negative /
    n_positive); the Random Forest uses ``class_weight="balanced"`` for the same
    reason.
    """
    from sklearn.pipeline import Pipeline

    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    xgb = XGBClassifier(
        n_estimators=xgb_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )
    gb = GradientBoostingClassifier(random_state=random_state)

    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", voting)])


def _scale_pos_weight(y) -> float:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return (neg / pos) if pos else 1.0


def train_ensemble(data: pd.DataFrame, seed: int = 42):
    """Train the ensemble and return ``(pipeline, metrics)``."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    model = build_ensemble(scale_pos_weight=_scale_pos_weight(y_train), random_state=seed)
    model.fit(X_train, y_train)
    # Ensemble has no single feature_importances_; skip that part of evaluate().
    metrics = _evaluate_no_importances(model, X_test, y_test)
    return model, metrics


def _evaluate_no_importances(model, X_test, y_test) -> dict:
    """Like evaluate() but without feature importances (VotingClassifier lacks them)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FraudShield ensemble")
    parser.add_argument("--data", default=None)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--model-out", default=DEFAULT_ENSEMBLE_PATH)
    parser.add_argument("--metrics-out", default="models/ensemble_metrics.json")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        data = pd.read_csv(args.data)
    else:
        data = generate_transactions(n_samples=args.n, random_state=args.seed)

    print("Training ensemble (RandomForest + XGBoost + GradientBoosting)...")
    model, metrics = train_ensemble(data, seed=args.seed)

    print("\n=== Ensemble evaluation ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:<10}: {metrics[key]:.4f}")

    # For context, show how the single Random Forest does on the same data.
    rf_model, rf_metrics = _single_rf_baseline(data, args.seed)
    print("\n(For comparison) single Random Forest ROC-AUC: "
          f"{rf_metrics['roc_auc']:.4f} | ensemble: {metrics['roc_auc']:.4f}")

    if args.save:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(model, args.model_out)
        with open(args.metrics_out, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nSaved ensemble -> {args.model_out}")


def _single_rf_baseline(data: pd.DataFrame, seed: int):
    from .train import train

    return train(data, n_estimators=200, seed=seed)


if __name__ == "__main__":
    main()
