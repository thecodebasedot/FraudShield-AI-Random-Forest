"""Kafka streaming pipeline for FraudShield AI.

Scores transactions off a Kafka topic in real time — the shape a high-throughput
bank deployment needs. Three modes:

  * ``produce``   — publish (synthetic) transactions to a Kafka topic
  * ``consume``   — read the topic, score each transaction, persist + alert
  * ``simulate``  — run the whole producer→scorer→sink pipeline in-process with
                    NO broker, so you can see it work anywhere (used by tests)

Kafka modes require a running broker (see docker-compose.yml). Configure with
``KAFKA_BOOTSTRAP_SERVERS`` (default ``localhost:9092``) and ``KAFKA_TOPIC``
(default ``transactions``).

Usage::

    python -m src.streaming simulate --n 50
    python -m src.streaming produce --n 1000
    python -m src.streaming consume
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable, Iterator

from .data_generator import generate_transactions
from .model import FEATURE_COLUMNS
from .predict import DEFAULT_MODEL_PATH, FraudDetector

DEFAULT_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TOPIC = os.environ.get("KAFKA_TOPIC", "transactions")


def transaction_stream(n: int, seed: int = 7) -> Iterator[dict[str, Any]]:
    """Yield ``n`` synthetic transactions as plain dicts (feature columns only)."""
    df = generate_transactions(n_samples=n, random_state=seed)[FEATURE_COLUMNS]
    for record in df.to_dict(orient="records"):
        yield record


# --------------------------------------------------------------------------- #
# Sink: what to do with each scored transaction
# --------------------------------------------------------------------------- #
def _sink(transaction: dict[str, Any], verdict: dict[str, Any], persist: bool) -> None:
    """Persist + alert for a scored transaction (best effort)."""
    if persist:
        try:
            from . import alerts, db

            db.init_db()
            db.record_prediction(transaction, verdict, api_key_name="stream")
            alerts.send_alert(transaction, verdict)
        except Exception:
            pass


def score_stream(
    transactions: Iterable[dict[str, Any]],
    detector: FraudDetector,
    persist: bool = True,
    use_cache: bool = True,
) -> dict[str, int]:
    """Score an iterable of transactions; returns a small run summary."""
    cache = None
    if use_cache:
        from .cache import PredictionCache, cached_score

        cache = PredictionCache()

    summary = {"processed": 0, "fraud": 0, "high_risk": 0, "cache_hits": 0}
    for txn in transactions:
        if cache is not None:
            verdict = cached_score(detector, cache, txn)
            summary["cache_hits"] += int(verdict.get("cached", False))
        else:
            verdict = detector.score(txn)

        summary["processed"] += 1
        summary["fraud"] += int(verdict["is_fraud"])
        summary["high_risk"] += int(verdict["risk_level"] == "HIGH")
        _sink(txn, verdict, persist)
    return summary


# --------------------------------------------------------------------------- #
# Kafka producer / consumer (require a live broker)
# --------------------------------------------------------------------------- #
def produce(n: int, topic: str = DEFAULT_TOPIC, bootstrap: str = DEFAULT_BOOTSTRAP, seed: int = 7) -> int:
    """Publish ``n`` synthetic transactions to Kafka. Returns count produced."""
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    count = 0
    for txn in transaction_stream(n, seed=seed):
        producer.send(topic, txn)
        count += 1
    producer.flush()
    producer.close()
    return count


def consume(
    detector: FraudDetector,
    topic: str = DEFAULT_TOPIC,
    bootstrap: str = DEFAULT_BOOTSTRAP,
    max_messages: int | None = None,
    persist: bool = True,
) -> dict[str, int]:
    """Consume the topic and score each transaction until ``max_messages``."""
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        consumer_timeout_ms=10_000,
    )

    def _messages() -> Iterator[dict[str, Any]]:
        for i, message in enumerate(consumer):
            if max_messages is not None and i >= max_messages:
                break
            yield message.value

    try:
        return score_stream(_messages(), detector, persist=persist)
    finally:
        consumer.close()


def simulate(n: int, model_path: str = DEFAULT_MODEL_PATH, persist: bool = False, seed: int = 7) -> dict[str, int]:
    """Run the full pipeline in-process with no broker. Great for demos/tests."""
    detector = FraudDetector(model_path=model_path)
    return score_stream(transaction_stream(n, seed=seed), detector, persist=persist)


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudShield Kafka streaming")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_sim = sub.add_parser("simulate", help="run pipeline with no broker")
    p_sim.add_argument("--n", type=int, default=50)
    p_sim.add_argument("--persist", action="store_true")

    p_prod = sub.add_parser("produce", help="publish transactions to Kafka")
    p_prod.add_argument("--n", type=int, default=1000)

    p_cons = sub.add_parser("consume", help="consume + score from Kafka")
    p_cons.add_argument("--max", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "simulate":
        summary = simulate(args.n, persist=args.persist)
        print(json.dumps(summary, indent=2))
    elif args.mode == "produce":
        count = produce(args.n)
        print(f"Produced {count} transactions to topic '{DEFAULT_TOPIC}'")
    elif args.mode == "consume":
        detector = FraudDetector()
        summary = consume(detector, max_messages=args.max)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
