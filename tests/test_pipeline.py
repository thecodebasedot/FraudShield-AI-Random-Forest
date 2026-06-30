"""Smoke tests for the FraudShield AI pipeline.

Run with:  python -m pytest -q   (or)   python -m unittest -v
These keep the dataset tiny so the whole suite trains in a couple of seconds.
"""

import unittest

from src.data_generator import generate_transactions
from src.model import FEATURE_COLUMNS, TARGET_COLUMN
from src.predict import FraudDetector
from src.train import train


class TestDataGenerator(unittest.TestCase):
    def test_shape_and_label(self):
        df = generate_transactions(n_samples=1000, fraud_ratio=0.1, random_state=0)
        self.assertEqual(len(df), 1000)
        self.assertIn(TARGET_COLUMN, df.columns)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)
        # Fraud ratio should be roughly what we asked for.
        self.assertAlmostEqual(df[TARGET_COLUMN].mean(), 0.1, delta=0.02)

    def test_reproducible(self):
        a = generate_transactions(500, 0.05, random_state=7)
        b = generate_transactions(500, 0.05, random_state=7)
        self.assertTrue(a.equals(b))


class TestTrainingAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate_transactions(n_samples=3000, fraud_ratio=0.08, random_state=1)
        cls.model, cls.metrics = train(data, n_estimators=60, seed=1)

    def test_metrics_reasonable(self):
        # On this separable synthetic data the model should clear a low bar.
        self.assertGreater(self.metrics["roc_auc"], 0.85)
        self.assertGreater(self.metrics["recall"], 0.5)

    def test_detector_scores_transaction(self):
        detector = FraudDetector.__new__(FraudDetector)
        detector.model = self.model
        detector.threshold = 0.5

        risky = detector.score(
            {
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
        )
        self.assertIn("fraud_probability", risky)
        self.assertIn(risky["risk_level"], {"MINIMAL", "LOW", "MEDIUM", "HIGH"})
        self.assertIsInstance(risky["is_fraud"], bool)

    def test_missing_field_raises(self):
        detector = FraudDetector.__new__(FraudDetector)
        detector.model = self.model
        detector.threshold = 0.5
        with self.assertRaises(ValueError):
            detector.score({"amount": 10.0})


if __name__ == "__main__":
    unittest.main()
