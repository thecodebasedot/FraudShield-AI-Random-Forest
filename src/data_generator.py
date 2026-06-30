"""Synthetic transaction data generator for FraudShield AI.

Real bank/online transaction datasets are sensitive and rarely shareable, so
this module fabricates a realistic-looking dataset with the kind of signals a
fraud model would learn from. The generated data is *only* for development,
demos and tests — it is not real financial data.

Fraud is injected with patterns that loosely mirror real life:
  * unusually large amounts,
  * transactions at odd hours (late night),
  * high recent transaction velocity,
  * foreign / mismatched country,
  * brand-new account / device.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Categorical vocabularies used across the project.
MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "entertainment",
    "restaurant",
    "fuel",
    "online_retail",
    "cash_withdrawal",
    "money_transfer",
]

DEVICE_TYPES = ["mobile", "web", "pos", "atm"]


def generate_transactions(
    n_samples: int = 20_000,
    fraud_ratio: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset.

    Parameters
    ----------
    n_samples:
        Total number of transactions to generate.
    fraud_ratio:
        Approximate fraction of transactions that are fraudulent.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame with raw transaction columns plus an ``is_fraud`` label.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = int(round(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    legit = _generate_legit(n_legit, rng)
    fraud = _generate_fraud(n_fraud, rng)

    df = pd.concat([legit, fraud], ignore_index=True)

    # Inject a little label noise so the classes aren't perfectly separable —
    # real fraud detection is never 100% clean, and this keeps the demo honest.
    flip = rng.random(len(df)) < 0.015
    df.loc[flip, "is_fraud"] = 1 - df.loc[flip, "is_fraud"]

    # Shuffle so fraud isn't clustered at the end.
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1))
    return df


def _generate_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the well-behaved majority class."""
    amount = np.round(rng.gamma(shape=2.0, scale=40.0, size=n) + 1.0, 2)

    # Legit activity concentrates during waking hours.
    hour = rng.integers(6, 23, size=n)

    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n),
            "device_type": rng.choice(
                DEVICE_TYPES, size=n, p=[0.5, 0.3, 0.15, 0.05]
            ),
            # Velocity: transactions by this customer in the last hour.
            "txn_count_1h": rng.poisson(lam=1.5, size=n),
            # Transactions in the last 24h.
            "txn_count_24h": rng.poisson(lam=8.0, size=n),
            # 0 = same country as account, 1 = foreign.
            "foreign_transaction": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
            # Age of the account in days.
            "account_age_days": rng.integers(20, 3650, size=n),
            # Whether this device has been seen before for this account.
            "is_new_device": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "is_fraud": 0,
        }
    )


def _generate_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the fraudulent minority class with shifted distributions."""
    # Fraudulent amounts skew larger, with a fat tail — but plenty of fraud is
    # small "card-testing" activity too, so the low end overlaps with legit.
    amount = np.round(rng.gamma(shape=2.0, scale=90.0, size=n) + 5.0, 2)

    # Fraud leans toward the night (0-5) and evening, but not exclusively.
    hour = np.where(
        rng.random(n) < 0.55,
        rng.integers(0, 6, size=n),
        rng.integers(8, 24, size=n),
    )

    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": rng.choice(
                MERCHANT_CATEGORIES,
                size=n,
                # Fraud leans on cash / transfers / high-value goods.
                p=[0.05, 0.18, 0.12, 0.05, 0.05, 0.05, 0.15, 0.17, 0.18],
            ),
            "device_type": rng.choice(
                DEVICE_TYPES, size=n, p=[0.35, 0.45, 0.05, 0.15]
            ),
            "txn_count_1h": rng.poisson(lam=3.5, size=n),
            "txn_count_24h": rng.poisson(lam=13.0, size=n),
            "foreign_transaction": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
            "account_age_days": rng.integers(0, 900, size=n),
            "is_new_device": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
            "is_fraud": 1,
        }
    )


if __name__ == "__main__":  # pragma: no cover - manual utility
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic transactions")
    parser.add_argument("--n", type=int, default=20_000, help="number of samples")
    parser.add_argument("--fraud-ratio", type=float, default=0.04)
    parser.add_argument("--out", default="data/transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = generate_transactions(args.n, args.fraud_ratio, args.seed)
    data.to_csv(args.out, index=False)
    print(
        f"Wrote {len(data):,} transactions "
        f"({data['is_fraud'].mean():.1%} fraud) to {args.out}"
    )
