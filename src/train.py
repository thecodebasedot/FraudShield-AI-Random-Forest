"""Train and persist the FraudShield AI Random Forest model.

Usage
-----
    python -m src.train                 # generate data on the fly + train
    python -m src.train --data data/transactions.csv
    python -m src.train --n 50000 --estimators 300
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_generator import generate_transactions
from .evaluate import evaluate
from .model import FEATURE_COLUMNS, TARGET_COLUMN, build_model

DEFAULT_MODEL_PATH = "models/fraudshield_rf.joblib"
DEFAULT_METRICS_PATH = "models/metrics.json"


def load_or_generate(data_path: str | None, n: int, fraud_ratio: float, seed: int) -> pd.DataFrame:
    """Load a CSV if provided/existing, otherwise generate synthetic data."""
    if data_path and os.path.exists(data_path):
        print(f"Loading transactions from {data_path}")
        return pd.read_csv(data_path)
    print(f"Generating {n:,} synthetic transactions ({fraud_ratio:.1%} fraud)")
    return generate_transactions(n_samples=n, fraud_ratio=fraud_ratio, random_state=seed)


def train(
    data: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int | None = None,
    test_size: float = 0.25,
    seed: int = 42,
):
    """Train the model and return ``(pipeline, metrics)``."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = build_model(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    print(f"Training Random Forest on {len(X_train):,} transactions...")
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FraudShield AI model")
    parser.add_argument("--data", default=None, help="optional CSV of transactions")
    parser.add_argument("--n", type=int, default=20_000, help="samples if generating")
    parser.add_argument("--fraud-ratio", type=float, default=0.04)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    data = load_or_generate(args.data, args.n, args.fraud_ratio, args.seed)
    model, metrics = train(
        data,
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    print("\n=== Evaluation (held-out test set) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 score : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    print("\nTop features by importance:")
    for name, imp in metrics["top_features"]:
        print(f"  {name:<22} {imp:.4f}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(model, args.model_out)
    with open(args.metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved model   -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
