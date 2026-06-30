"""Tests for the SHAP explainer and the real-dataset trainer."""

import unittest

import numpy as np
import pandas as pd

from src.data_generator import generate_transactions
from src.explain import TransactionExplainer
from src.predict import FraudDetector
from src.realdata import train_numeric
from src.train import train

RISKY_TXN = {
    "amount": 2000.0,
    "hour": 3,
    "txn_count_1h": 9,
    "txn_count_24h": 30,
    "foreign_transaction": 1,
    "account_age_days": 5,
    "is_new_device": 1,
    "merchant_category": "money_transfer",
    "device_type": "web",
}


class TestExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate_transactions(n_samples=3000, fraud_ratio=0.08, random_state=2)
        cls.model, _ = train(data, n_estimators=60, seed=2)

    def _explainer(self) -> TransactionExplainer:
        # Build without touching disk by injecting the in-memory model.
        exp = TransactionExplainer.__new__(TransactionExplainer)
        exp.detector = FraudDetector.__new__(FraudDetector)
        exp.detector.model = self.model
        exp.detector.threshold = 0.5
        exp.model = self.model
        exp.preprocessor = self.model.named_steps["preprocess"]
        exp.classifier = self.model.named_steps["classifier"]
        exp.feature_names = exp._expanded_feature_names()
        exp._explainer = None
        return exp

    def test_explanation_structure(self):
        result = self._explainer().explain(RISKY_TXN, top_k=5)
        self.assertIn("fraud_probability", result)
        self.assertIn("explanation", result)
        self.assertLessEqual(len(result["explanation"]), 5)
        for item in result["explanation"]:
            self.assertIn(item["direction"], {"increases_risk", "decreases_risk"})
        # Each reason should map to a positive (risk-increasing) contribution.
        self.assertIsInstance(result["reasons"], list)


class TestRealDataTrainer(unittest.TestCase):
    def test_train_numeric_on_kaggle_like_schema(self):
        rng = np.random.default_rng(0)
        n = 2000
        df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
        df["Amount"] = np.round(rng.gamma(2, 50, n), 2)
        score = 1.5 * df["V1"] + 0.01 * df["Amount"] + rng.normal(0, 1, n)
        y = (score > np.quantile(score, 0.95)).astype(int)

        model, metrics = train_numeric(df, y, n_estimators=50, seed=0)
        self.assertGreater(metrics["roc_auc"], 0.8)
        self.assertEqual(metrics["test_size"], int(round(n * 0.25)))


if __name__ == "__main__":
    unittest.main()
