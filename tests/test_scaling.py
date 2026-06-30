"""Tests for Phase-2 scaling features: ensemble, cache and streaming pipeline.

Kafka/Redis servers are never required — the cache uses its in-memory fallback
and streaming runs in simulation mode.
"""

import unittest

from src.cache import PredictionCache, cached_score, make_key
from src.data_generator import generate_transactions
from src.ensemble import train_ensemble
from src.predict import FraudDetector
from src.streaming import score_stream, simulate, transaction_stream
from src.train import train

RISKY = {
    "amount": 1450.0, "hour": 3, "txn_count_1h": 7, "txn_count_24h": 25,
    "foreign_transaction": 1, "account_age_days": 12, "is_new_device": 1,
    "merchant_category": "money_transfer", "device_type": "web",
}


def _detector_from(model):
    d = FraudDetector.__new__(FraudDetector)
    d.model = model
    d.threshold = 0.5
    return d


class TestEnsemble(unittest.TestCase):
    def test_train_ensemble_reasonable(self):
        data = generate_transactions(n_samples=4000, fraud_ratio=0.08, random_state=3)
        model, metrics = train_ensemble(data, seed=3)
        self.assertGreater(metrics["roc_auc"], 0.8)
        # The pipeline must expose predict_proba so FraudDetector can use it.
        detector = _detector_from(model)
        verdict = detector.score(RISKY)
        self.assertIn(verdict["risk_level"], {"MINIMAL", "LOW", "MEDIUM", "HIGH"})


class TestCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate_transactions(n_samples=3000, fraud_ratio=0.08, random_state=4)
        cls.model, _ = train(data, n_estimators=60, seed=4)

    def test_key_is_stable_and_field_order_independent(self):
        reordered = dict(reversed(list(RISKY.items())))
        self.assertEqual(make_key(RISKY), make_key(reordered))

    def test_miss_then_hit(self):
        cache = PredictionCache()  # no REDIS_URL -> memory backend
        self.assertEqual(cache.backend, "memory")
        detector = _detector_from(self.model)

        first = cached_score(detector, cache, RISKY)
        second = cached_score(detector, cache, RISKY)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["fraud_probability"], second["fraud_probability"])

    def test_unreachable_redis_falls_back(self):
        # Nothing is listening here; must transparently degrade to memory.
        cache = PredictionCache(redis_url="redis://127.0.0.1:6390/0")
        self.assertEqual(cache.backend, "memory")


class TestStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate_transactions(n_samples=3000, fraud_ratio=0.08, random_state=5)
        cls.model, _ = train(data, n_estimators=60, seed=5)

    def test_transaction_stream_count_and_shape(self):
        items = list(transaction_stream(10, seed=1))
        self.assertEqual(len(items), 10)
        self.assertIn("amount", items[0])
        self.assertNotIn("is_fraud", items[0])  # features only, no label

    def test_score_stream_summary(self):
        detector = _detector_from(self.model)
        summary = score_stream(
            transaction_stream(100, seed=2), detector, persist=False, use_cache=True
        )
        self.assertEqual(summary["processed"], 100)
        self.assertGreaterEqual(summary["fraud"], 0)
        self.assertLessEqual(summary["fraud"], 100)

    def test_score_stream_caches_duplicates(self):
        detector = _detector_from(self.model)
        # Feed the same transaction many times -> all but the first are cache hits.
        summary = score_stream([RISKY] * 5, detector, persist=False, use_cache=True)
        self.assertEqual(summary["processed"], 5)
        self.assertEqual(summary["cache_hits"], 4)


if __name__ == "__main__":
    unittest.main()
