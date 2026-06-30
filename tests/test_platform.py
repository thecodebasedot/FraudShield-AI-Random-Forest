"""Tests for the Phase-1 platform: database, auth, alerts and API integration.

Each test points DATABASE_URL at a throwaway SQLite file so it never touches a
real database. db/auth modules are imported fresh per test class against that URL.
"""

import importlib
import os
import tempfile
import unittest


class _TempDbTestCase(unittest.TestCase):
    """Base case that gives each subclass its own SQLite file + fresh modules."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        os.environ["DATABASE_URL"] = f"sqlite:///{cls._tmp.name}"
        # Re-import so the engine binds to this URL.
        import src.db as db
        import src.auth as auth

        cls.db = importlib.reload(db)
        cls.auth = importlib.reload(auth)
        cls.db.init_db()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tmp.name)
        os.environ.pop("DATABASE_URL", None)

    def setUp(self):
        # Isolate every test method from the others (shared class DB).
        with self.db.session_scope() as session:
            for model in (self.db.Prediction, self.db.AuditLog, self.db.ApiKey):
                session.query(model).delete()


RISKY = {
    "amount": 1450.0, "hour": 3, "txn_count_1h": 7, "txn_count_24h": 25,
    "foreign_transaction": 1, "account_age_days": 12, "is_new_device": 1,
    "merchant_category": "money_transfer", "device_type": "web",
}
VERDICT_HIGH = {"fraud_probability": 0.88, "is_fraud": True, "risk_level": "HIGH"}
VERDICT_LOW = {"fraud_probability": 0.05, "is_fraud": False, "risk_level": "MINIMAL"}


class TestDatabase(_TempDbTestCase):
    def test_record_and_stats(self):
        self.db.record_prediction(RISKY, VERDICT_HIGH, api_key_name="t")
        self.db.record_prediction(RISKY, VERDICT_LOW, api_key_name="t")
        summary = self.db.stats_summary()
        self.assertEqual(summary["total_transactions"], 2)
        self.assertEqual(summary["fraud_flagged"], 1)
        self.assertAlmostEqual(summary["fraud_rate"], 0.5)
        self.assertEqual(summary["by_risk_level"].get("HIGH"), 1)

    def test_recent_predictions(self):
        self.db.record_prediction(RISKY, VERDICT_HIGH)
        recent = self.db.recent_predictions(limit=5)
        self.assertGreaterEqual(len(recent), 1)
        self.assertEqual(recent[0]["risk_level"], "HIGH")

    def test_audit_log(self):
        # Should not raise.
        self.db.record_audit(action="predict", api_key_name="t", detail="risk=HIGH")


class TestAuth(_TempDbTestCase):
    def test_key_lifecycle(self):
        raw = self.auth.create_api_key("client-a")
        self.assertTrue(raw.startswith("fsk_"))
        self.assertEqual(self.auth.verify_api_key(raw), "client-a")
        self.assertIsNone(self.auth.verify_api_key("fsk_bogus"))
        self.assertIsNone(self.auth.verify_api_key(None))

    def test_duplicate_name_rejected(self):
        self.auth.create_api_key("client-b")
        with self.assertRaises(ValueError):
            self.auth.create_api_key("client-b")

    def test_revoke(self):
        raw = self.auth.create_api_key("client-c")
        self.assertTrue(self.auth.revoke_api_key("client-c"))
        self.assertIsNone(self.auth.verify_api_key(raw))  # revoked -> invalid


class TestAlerts(_TempDbTestCase):
    def test_high_risk_alerts_via_log_fallback(self):
        from src import alerts

        # No Slack/SMTP configured -> falls back to log but still "handles" it.
        self.assertTrue(alerts.send_alert(RISKY, VERDICT_HIGH))

    def test_low_risk_does_not_alert(self):
        from src import alerts

        self.assertFalse(alerts.send_alert(RISKY, VERDICT_LOW))


if __name__ == "__main__":
    unittest.main()
