# 🛡️ FraudShield AI — Random Forest

Machine-learning system that detects **fraud in bank / online transactions**
using a **Random Forest** classifier.

> **Project:** FraudShield AI
> **Problem solved:** ব্যাংক/অনলাইন ট্রানজ্যাকশনে জালিয়াতি শনাক্ত করা (detecting fraud in bank / online transactions)
> **ML algorithm:** Random Forest

---

## ✨ What it does

Given a transaction's attributes (amount, time, velocity, device, location, etc.),
FraudShield AI returns a **fraud probability**, a **flag** (fraud / not fraud) and a
**risk level** (`MINIMAL` → `LOW` → `MEDIUM` → `HIGH`).

Because real transaction data is sensitive, the project ships with a built-in
**synthetic data generator** that fabricates realistic transactions with planted
fraud patterns, so you can train and demo the whole pipeline end-to-end without
any private data.

## 🧠 Why Random Forest?

- Handles the **mixed numeric + categorical** features of transactions well.
- Robust to outliers and naturally captures **non-linear interactions**
  (e.g. "large amount **and** new device **and** 3 a.m.").
- `class_weight="balanced"` copes with the heavy **class imbalance** (fraud is rare).
- Gives **feature importances** for explainability — important in finance.

## 📦 Project structure

```
FraudShield-AI-Random-Forest/
├── src/
│   ├── data_generator.py   # synthetic transaction dataset
│   ├── model.py            # preprocessing + RandomForest pipeline
│   ├── train.py            # train, evaluate, persist the model
│   ├── evaluate.py         # metrics + feature importances
│   ├── predict.py          # score new transactions (FraudDetector + CLI)
│   ├── api.py              # FastAPI REST service
│   └── visualize.py        # generate evaluation charts
├── tests/
│   └── test_pipeline.py    # smoke tests for data, training, prediction
├── reports/                # generated evaluation charts (PNG)
├── data/                   # generated CSVs (git-ignored)
├── models/                 # saved model + metrics (git-ignored)
└── requirements.txt
```

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates synthetic data automatically)
python -m src.train

# 3. Score a transaction (uses a built-in risky example)
python -m src.predict
```

### Train on your own CSV

The trainer accepts any CSV with these columns:

| column                | type       | meaning                                   |
|-----------------------|------------|-------------------------------------------|
| `amount`              | float      | transaction amount                        |
| `hour`                | int 0–23   | hour of day                               |
| `txn_count_1h`        | int        | customer's transactions in the last hour  |
| `txn_count_24h`       | int        | customer's transactions in the last 24h   |
| `foreign_transaction` | 0/1        | 1 if foreign / cross-border                |
| `account_age_days`    | int        | age of the account in days                |
| `is_new_device`       | 0/1        | 1 if device never seen before             |
| `merchant_category`   | category   | e.g. `money_transfer`, `electronics`      |
| `device_type`         | category   | `mobile` / `web` / `pos` / `atm`          |
| `is_fraud`            | 0/1        | label (training only)                     |

```bash
python -m src.data_generator --n 50000 --out data/transactions.csv
python -m src.train --data data/transactions.csv --estimators 300
```

### Score a custom transaction

```bash
python -m src.predict --json '{
  "amount": 25.0, "hour": 14, "txn_count_1h": 1, "txn_count_24h": 4,
  "foreign_transaction": 0, "account_age_days": 800, "is_new_device": 0,
  "merchant_category": "grocery", "device_type": "pos"
}'
```

### Use it from Python

```python
from src.predict import FraudDetector

detector = FraudDetector("models/fraudshield_rf.joblib")
verdict = detector.score({
    "amount": 1450.0, "hour": 3, "txn_count_1h": 7, "txn_count_24h": 25,
    "foreign_transaction": 1, "account_age_days": 12, "is_new_device": 1,
    "merchant_category": "money_transfer", "device_type": "web",
})
print(verdict)
# {'fraud_probability': 0.88, 'is_fraud': True, 'risk_level': 'HIGH'}
```

## 🌐 REST API

Serve the model over HTTP with **FastAPI**:

```bash
uvicorn src.api:app --reload
```

Then visit **http://127.0.0.1:8000/docs** for interactive Swagger UI.

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | model status & threshold             |
| POST   | `/predict`        | score a single transaction           |
| POST   | `/predict/batch`  | score a list of transactions         |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount":1450.0,"hour":3,"txn_count_1h":7,"txn_count_24h":25,
       "foreign_transaction":1,"account_age_days":12,"is_new_device":1,
       "merchant_category":"money_transfer","device_type":"web"}'
# {"fraud_probability":0.88,"is_fraud":true,"risk_level":"HIGH"}
```

Requests are validated (e.g. `hour` must be 0–23) and the model is loaded lazily,
so the service boots even before a model exists — it returns `503` until you train one.

## 📈 Visualization

Generate evaluation charts into `reports/`:

```bash
python -m src.visualize
```

| ROC curve | Confusion matrix |
|-----------|------------------|
| ![ROC curve](reports/roc_curve.png) | ![Confusion matrix](reports/confusion_matrix.png) |

| Feature importance | Fraud-score separation |
|--------------------|------------------------|
| ![Feature importance](reports/feature_importance.png) | ![Probability distribution](reports/probability_distribution.png) |

## 📊 Example results

On a synthetic test set (20k transactions, ~4% fraud) a typical run produces:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.97  |
| Precision | ~0.87  |
| Recall    | ~0.64  |
| F1        | ~0.74  |
| ROC AUC   | ~0.86  |

Top predictive features are usually `account_age_days`, `hour`, `amount` and
recent transaction velocity — the same signals human fraud analysts watch.

> Scores vary slightly per run because the data is randomly generated. The data
> includes deliberate label noise and class overlap so the model is **not**
> trivially perfect — just like real-world fraud detection.

## ✅ Running the tests

```bash
python -m unittest discover -s tests -v
```

## ⚠️ Disclaimer

This repository uses **synthetic data** and is intended for learning and
demonstration. It is **not** a production fraud system and should not be used to
make real financial decisions without proper data, validation and compliance review.

## 📄 License

See [LICENSE](LICENSE).
