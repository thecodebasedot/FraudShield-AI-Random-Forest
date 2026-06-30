"""Model definition for FraudShield AI.

Builds a scikit-learn ``Pipeline`` that one-hot encodes the categorical
transaction fields and feeds everything into a ``RandomForestClassifier``.
Keeping preprocessing inside the pipeline means the saved model can score raw
transaction dictionaries without the caller re-implementing feature encoding.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Columns the model consumes (everything except ids and the label).
NUMERIC_FEATURES = [
    "amount",
    "hour",
    "txn_count_1h",
    "txn_count_24h",
    "foreign_transaction",
    "account_age_days",
    "is_new_device",
]
CATEGORICAL_FEATURES = [
    "merchant_category",
    "device_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_fraud"


def build_model(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Construct the preprocessing + Random Forest pipeline.

    ``class_weight="balanced"`` makes the forest pay attention to the rare
    fraud class despite the heavy class imbalance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
