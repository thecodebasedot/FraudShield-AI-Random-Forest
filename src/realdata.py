"""Train FraudShield AI on a real, all-numeric fraud dataset.

The flagship public benchmark is Kaggle's *Credit Card Fraud Detection*
dataset (``creditcard.csv``): columns ``Time``, ``V1``..``V28``, ``Amount`` and
a ``Class`` label (1 = fraud). Its features are anonymized PCA components, so
they don't match the synthetic schema used elsewhere in this project — this
module therefore trains a *separate*, schema-agnostic numeric model.

Download (requires a Kaggle account)::

    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Then::

    python -m src.realdata --data creditcard.csv --target Class

Any CSV of numeric features plus a binary target works, not just Kaggle's.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_dataset(path: str, target: str, drop: list[str] | None = None):
    """Load a CSV and split into numeric feature matrix X and label y."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found. Download the Kaggle creditcard dataset or pass "
            "your own CSV via --data."
        )
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in CSV columns: {list(df.columns)}")

    drop = drop or []
    feature_cols = [c for c in df.columns if c != target and c not in drop]
    X = df[feature_cols].select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")
    y = df[target].astype(int)
    return X, y, list(X.columns)


def train_numeric(X, y, n_estimators: int = 200, seed: int = 42):
    """Train a balanced Random Forest on a numeric feature matrix."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_size": int(len(y_test)),
        "fraud_in_test": int(y_test.sum()),
    }
    return clf, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on a real numeric fraud dataset")
    parser.add_argument("--data", required=True, help="path to CSV (e.g. creditcard.csv)")
    parser.add_argument("--target", default="Class", help="label column name")
    parser.add_argument("--drop", nargs="*", default=None, help="columns to ignore")
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="models/fraudshield_real.joblib")
    parser.add_argument("--metrics-out", default="models/real_metrics.json")
    args = parser.parse_args()

    X, y, cols = load_dataset(args.data, args.target, args.drop)
    print(f"Loaded {len(X):,} rows, {len(cols)} numeric features, "
          f"{y.mean():.3%} fraud")

    model, metrics = train_numeric(X, y, n_estimators=args.estimators, seed=args.seed)
    print("\n=== Evaluation ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:<10}: {metrics[key]:.4f}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump({"model": model, "features": cols}, args.model_out)
    with open(args.metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved model   -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
