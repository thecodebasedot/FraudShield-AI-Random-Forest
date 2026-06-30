"""Tests for Phase-3 enterprise features: security helpers and multi-tenancy."""

import importlib
import os
import tempfile
import unittest

from src import security


class TestEncryption(unittest.TestCase):
    def test_roundtrip_with_key(self):
        os.environ["FRAUDSHIELD_ENC_KEY"] = security.generate_key()
        try:
            token = security.encrypt_field("sensitive-value")
            self.assertNotEqual(token, "sensitive-value")
            self.assertEqual(security.decrypt_field(token), "sensitive-value")
        finally:
            os.environ.pop("FRAUDSHIELD_ENC_KEY", None)

    def test_noop_without_key(self):
        os.environ.pop("FRAUDSHIELD_ENC_KEY", None)
        self.assertEqual(security.encrypt_field("plain"), "plain")
        self.assertEqual(security.decrypt_field("plain"), "plain")


class TestMasking(unittest.TestCase):
    def test_mask_pan(self):
        masked = security.mask_pan("card 4111 1111 1111 1111 used")
        self.assertIn("1111", masked)
        self.assertNotIn("4111 1111 1111 1111", masked)
        self.assertIn("*", masked)

    def test_mask_sensitive_redacts_secrets(self):
        safe = security.mask_sensitive({"password": "hunter2", "amount": 10})
        self.assertEqual(safe["password"], "***REDACTED***")
        self.assertEqual(safe["amount"], 10)


class TestRateLimiter(unittest.TestCase):
    def test_allows_then_blocks(self):
        limiter = security.RateLimiter(per_minute=2)
        self.assertTrue(limiter.allow("client-x"))
        self.assertTrue(limiter.allow("client-x"))
        self.assertFalse(limiter.allow("client-x"))  # bucket empty

    def test_isolated_per_client(self):
        limiter = security.RateLimiter(per_minute=1)
        self.assertTrue(limiter.allow("a"))
        self.assertTrue(limiter.allow("b"))  # different client, own bucket


class TestMultiTenant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        os.environ["DATABASE_URL"] = f"sqlite:///{cls._tmp.name}"
        import src.db as db
        import src.auth as auth

        cls.db = importlib.reload(db)
        cls.auth = importlib.reload(auth)
        cls.db.init_db()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tmp.name)
        os.environ.pop("DATABASE_URL", None)

    def test_keys_carry_tenant(self):
        self.auth.create_api_key("key-a", tenant="bank-a")
        principal = self.auth.resolve_api_key(
            # re-create to get raw key
            self._fresh_key("key-a2", "bank-a")
        )
        self.assertEqual(principal["tenant"], "bank-a")

    def _fresh_key(self, name, tenant):
        return self.auth.create_api_key(name, tenant=tenant)

    def test_stats_scoped_by_tenant(self):
        txn = {
            "amount": 100.0, "hour": 2, "txn_count_1h": 5, "txn_count_24h": 20,
            "foreign_transaction": 1, "account_age_days": 10, "is_new_device": 1,
            "merchant_category": "money_transfer", "device_type": "web",
        }
        high = {"fraud_probability": 0.9, "is_fraud": True, "risk_level": "HIGH"}
        low = {"fraud_probability": 0.1, "is_fraud": False, "risk_level": "MINIMAL"}

        self.db.record_prediction(txn, high, tenant="acme")
        self.db.record_prediction(txn, low, tenant="acme")
        self.db.record_prediction(txn, high, tenant="globex")

        acme = self.db.stats_summary(tenant="acme")
        globex = self.db.stats_summary(tenant="globex")
        self.assertEqual(acme["total_transactions"], 2)
        self.assertEqual(acme["fraud_flagged"], 1)
        self.assertEqual(globex["total_transactions"], 1)

        # Tenant-scoped recent predictions never leak across tenants.
        acme_recent = self.db.recent_predictions(tenant="acme")
        self.assertEqual(len(acme_recent), 2)


if __name__ == "__main__":
    unittest.main()
