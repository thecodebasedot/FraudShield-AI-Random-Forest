"""Generate evaluation charts for FraudShield AI.

Trains (or loads) the model, then renders a set of PNGs that explain how the
fraud detector behaves:

  * confusion_matrix.png      — true/false positives & negatives
  * roc_curve.png             — ROC curve with AUC
  * feature_importance.png    — what the forest relies on
  * probability_distribution.png — fraud-score separation between classes

Charts are written to ``reports/`` and are safe to run headless (Agg backend).

Usage::

    python -m src.visualize                 # generate data + train + plot
    python -m src.visualize --data data/transactions.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split

from .data_generator import generate_transactions
from .evaluate import _feature_importances
from .model import FEATURE_COLUMNS, TARGET_COLUMN, build_model

REPORTS_DIR = "reports"


def _load_data(data_path: str | None, n: int, seed: int):
    if data_path and os.path.exists(data_path):
        import pandas as pd

        return pd.read_csv(data_path)
    return generate_transactions(n_samples=n, random_state=seed)


def plot_confusion_matrix(model, X_test, y_test, out: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["legit", "fraud"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, out: str) -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, out: str) -> None:
    importances = _feature_importances(model, top_k=12)
    names = [n for n, _ in importances][::-1]
    values = [v for _, v in importances][::-1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names, values, color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_probability_distribution(model, X_test, y_test, out: str) -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    legit = y_proba[np.asarray(y_test) == 0]
    fraud = y_proba[np.asarray(y_test) == 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(legit, bins=bins, alpha=0.6, label="legit", color="#1f77b4", density=True)
    ax.hist(fraud, bins=bins, alpha=0.6, label="fraud", color="#d62728", density=True)
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Density")
    ax.set_title("Fraud-score separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FraudShield AI charts")
    parser.add_argument("--data", default=None, help="optional CSV of transactions")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=REPORTS_DIR)
    args = parser.parse_args()

    data = _load_data(args.data, args.n, args.seed)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )

    model = build_model(n_estimators=args.estimators, random_state=args.seed)
    print(f"Training on {len(X_train):,} transactions for visualization...")
    model.fit(X_train, y_train)

    os.makedirs(args.out_dir, exist_ok=True)
    charts = {
        "confusion_matrix.png": lambda p: plot_confusion_matrix(model, X_test, y_test, p),
        "roc_curve.png": lambda p: plot_roc_curve(model, X_test, y_test, p),
        "feature_importance.png": lambda p: plot_feature_importance(model, p),
        "probability_distribution.png": lambda p: plot_probability_distribution(
            model, X_test, y_test, p
        ),
    }
    for name, fn in charts.items():
        path = os.path.join(args.out_dir, name)
        fn(path)
        print(f"  wrote {path}")

    print(f"\nDone. {len(charts)} charts in {args.out_dir}/")


if __name__ == "__main__":
    main()
