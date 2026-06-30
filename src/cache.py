"""Caching layer for FraudShield AI.

Identical transactions (same features) get the same verdict, so caching scores
avoids recomputing them — valuable under bursty, repetitive traffic. Uses Redis
when ``REDIS_URL`` is set and reachable, and transparently falls back to a small
in-process LRU cache otherwise, so the code path is always safe.

Environment
-----------
  REDIS_URL          e.g. redis://localhost:6379/0  (optional)
  CACHE_TTL_SECONDS  cache entry lifetime (default 300)
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any

from .model import FEATURE_COLUMNS

DEFAULT_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
_KEY_PREFIX = "fraudshield:score:"


def make_key(transaction: dict[str, Any]) -> str:
    """Stable cache key from the model-relevant fields of a transaction."""
    payload = {k: transaction.get(k) for k in FEATURE_COLUMNS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode()).hexdigest()  # noqa: S324 - non-crypto use
    return _KEY_PREFIX + digest


class _MemoryCache:
    """Tiny bounded LRU used when Redis isn't available."""

    def __init__(self, maxsize: int = 10_000):
        self.maxsize = maxsize
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        # TTL is ignored in the in-memory fallback; the LRU bound caps growth.
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)


class PredictionCache:
    """Cache of transaction verdicts, Redis-backed with in-memory fallback."""

    def __init__(self, redis_url: str | None = None, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self.backend = "memory"
        self._redis = None
        self._memory = _MemoryCache()

        redis_url = redis_url or os.environ.get("REDIS_URL")
        if redis_url:
            self._try_connect_redis(redis_url)

    def _try_connect_redis(self, redis_url: str) -> None:
        try:
            import redis  # imported lazily

            client = redis.from_url(redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            # Unreachable / not installed -> silently use the in-memory fallback.
            self._redis = None
            self.backend = "memory"

    def get(self, transaction: dict[str, Any]) -> dict[str, Any] | None:
        key = make_key(transaction)
        raw = self._raw_get(key)
        return json.loads(raw) if raw else None

    def set(self, transaction: dict[str, Any], verdict: dict[str, Any]) -> None:
        key = make_key(transaction)
        self._raw_set(key, json.dumps(verdict))

    def _raw_get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                value = self._redis.get(key)
                return value.decode() if value else None
            except Exception:
                self._fallback()
        return self._memory.get(key)

    def _raw_set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                self._redis.set(key, value, ex=self.ttl)
                return
            except Exception:
                self._fallback()
        self._memory.set(key, value, ttl=self.ttl)

    def _fallback(self) -> None:
        """Drop to in-memory after a Redis error so we don't keep retrying."""
        self._redis = None
        self.backend = "memory"


def cached_score(detector, cache: PredictionCache, transaction: dict[str, Any]) -> dict[str, Any]:
    """Score a transaction, using and populating the cache."""
    hit = cache.get(transaction)
    if hit is not None:
        hit = dict(hit)
        hit["cached"] = True
        return hit
    verdict = detector.score(transaction)
    cache.set(transaction, verdict)
    result = dict(verdict)
    result["cached"] = False
    return result
