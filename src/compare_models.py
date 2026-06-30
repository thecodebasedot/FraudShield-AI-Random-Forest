"""Compare several classifiers and tune the Random Forest for FraudShield AI.

Even though the project ships with a Random Forest, it is worth showing *why*
that choice is reasonable. This module:

  1. Cross-validates a handful of models on the same data and reports ROC-AUC.
  2. Runs a randomized hyperparameter search over the Random Forest.
  3. Optionally retrains the best configuration on the full data and saves it.

Usage::

    python -m src.compare_models                 # compare + tune (synthetic data)
    python -m src.compare_models --data data/transactions.csv
    python -m src.compare_models --save          # persist the tuned best model
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_generator import generate_transactions
from .model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def _preprocessor(scale: bool = False) -> ColumnTransformer:
    """Shared preprocessing; linear models also get the numerics scaled."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    """The classifiers we put head-to-head."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("preprocess", _preprocessor(scale=True)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocess", _preprocessor()),
                ("classifier", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


def compare(X, y, cv: int = 5, seed: int = 42) -> list[dict]:
    """Cross-validate each candidate and return ROC-AUC stats, best first."""
    results = []
    for name, pipe in candidate_models(seed).items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append(
            {
                "model": name,
                "roc_auc_mean": float(scores.mean()),
                "roc_auc_std": float(scores.std()),
            }
        )
        print(f"  {name:<20} ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    results.sort(key=lambda r: r["roc_auc_mean"], reverse=True)
    return results


def tune_random_forest(X, y, n_iter: int = 20, cv: int = 4, seed: int = 42):
    """Randomized search over Random Forest hyperparameters."""
    pipe = Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced", n_jobs=-1, random_state=seed
                ),
            ),
        ]
    )
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [None, 6, 10, 16, 24],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def _load_data(data_path: str | None, n: int, seed: int) -> pd.DataFrame:
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    return generate_transactions(n_samples=n, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare & tune FraudShield models")
    parser.add_argument("--data", default=None)
    parser.add_argument("--n", type=int, default=12_000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20, help="tuning iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="persist tuned best model")
    parser.add_argument("--model-out", default="models/fraudshield_rf.joblib")
    parser.add_argument("--report-out", default="models/comparison.json")
    args = parser.parse_args()

    data = _load_data(args.data, args.n, args.seed)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    print(f"=== Model comparison ({args.cv}-fold CV, {len(X):,} rows) ===")
    ranking = compare(X, y, cv=args.cv, seed=args.seed)
    print(f"\nBest by ROC-AUC: {ranking[0]['model']}")

    print(f"\n=== Tuning Random Forest ({args.n_iter} iterations) ===")
    search = tune_random_forest(X, y, n_iter=args.n_iter, cv=max(2, args.cv - 1), seed=args.seed)
    print(f"Best CV ROC-AUC : {search.best_score_:.4f}")
    print("Best parameters :")
    for k, v in search.best_params_.items():
        print(f"  {k.replace('classifier__', '')}: {v}")

    report = {
        "comparison": ranking,
        "tuned_random_forest": {
            "best_score": float(search.best_score_),
            "best_params": {
                k.replace("classifier__", ""): v for k, v in search.best_params_.items()
            },
        },
    }
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved comparison report -> {args.report_out}")

    if args.save:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        joblib.dump(search.best_estimator_, args.model_out)
        print(f"Saved tuned best model  -> {args.model_out}")


if __name__ == "__main__":
    main()
