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
│   ├── api.py              # FastAPI REST service (auth, persistence, alerts)
│   ├── db.py               # SQLAlchemy persistence (predictions, audit, keys)
│   ├── auth.py             # API-key authentication
│   ├── alerts.py           # real-time Slack / email alerts
│   ├── dashboard.py        # Streamlit dashboard (+ admin analytics)
│   ├── compare_models.py   # model comparison + hyperparameter tuning
│   ├── explain.py          # SHAP per-transaction explainability
│   ├── realdata.py         # train on a real numeric dataset (e.g. Kaggle)
│   └── visualize.py        # generate evaluation charts
├── notebooks/
│   └── eda.ipynb           # exploratory data analysis
├── tests/
│   ├── test_pipeline.py    # data, training, prediction
│   ├── test_features.py    # SHAP explainer + real-dataset trainer
│   └── test_platform.py    # database, auth and alerts
├── .github/workflows/      # GitHub Actions CI
├── reports/                # generated evaluation charts (PNG)
├── data/                   # generated CSVs (git-ignored)
├── models/                 # saved model + metrics (git-ignored)
├── Dockerfile              # container image (serves the API)
├── docker-compose.yml
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

## 🎛️ Interactive dashboard

A **Streamlit** UI to score transactions in the browser — single transaction
form *and* batch CSV upload with a downloadable scored file.

```bash
streamlit run src/dashboard.py
```

Opens at http://localhost:8501. Adjust the decision threshold live from the sidebar.

## 🤖 Model comparison & tuning

FraudShield ships with a Random Forest, but you can verify that choice and tune it:

```bash
python -m src.compare_models           # 5-fold CV across models + RF tuning
python -m src.compare_models --save    # persist the tuned best model
```

It cross-validates **LogisticRegression**, **RandomForest** and **GradientBoosting**
on ROC-AUC, then runs a `RandomizedSearchCV` over the forest's hyperparameters and
writes the ranking to `models/comparison.json`. A typical run:

```
LogisticRegression   ROC-AUC = 0.851
RandomForest         ROC-AUC = 0.854   <- best
GradientBoosting     ROC-AUC = 0.853
Tuned RandomForest   ROC-AUC = 0.865
```

## 🏗️ Production platform (auth · persistence · audit · alerts)

FraudShield is more than a model — it ships the operational layer a real
deployment needs.

### 🔑 API-key authentication

The API runs in **open mode** until you create the first key, then locks down —
protected endpoints require an `X-API-Key` header. Only a SHA-256 hash is stored.

```bash
python -m src.auth create --name acme-bank   # prints the key once
python -m src.auth list
python -m src.auth revoke --name acme-bank

curl -H "X-API-Key: fsk_..." -X POST http://127.0.0.1:8000/predict -d '{...}'
```

### 🗄️ Persistence & audit trail

Every prediction and API action is stored via **SQLAlchemy** — SQLite by default,
PostgreSQL in production by pointing `DATABASE_URL` at it:

```bash
export DATABASE_URL=postgresql+psycopg://user:pass@host/fraudshield
```

Tables: `predictions` (scored transactions), `audit_logs` (activity trail), `api_keys`.

### 🔔 Real-time alerts

High-risk transactions trigger a **Slack** and/or **email** alert (with a log
fallback so nothing is ever lost). All optional, configured via env vars:

| Variable | Purpose |
|----------|---------|
| `FRAUDSHIELD_ALERT_LEVEL` | min risk to alert on (default `HIGH`) |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | email server |
| `ALERT_EMAIL_FROM` / `ALERT_EMAIL_TO` | email sender / recipients |

### 📊 Admin analytics

The dashboard's **Admin analytics** tab (and the `/stats` + `/predictions`
endpoints) show live totals, fraud rate, risk-level breakdown and the most
recent scored transactions — all read from the database.

## 🔍 Explainability (SHAP)

Every flag comes with a **reason**. SHAP attributes a prediction to individual
features so analysts know *why* a transaction was flagged:

```bash
python -m src.explain     # or POST /explain on the API
```

```json
{
  "fraud_probability": 0.88, "is_fraud": true, "risk_level": "HIGH",
  "reasons": ["young account", "unusual transaction hour",
              "high transaction amount", "high transaction velocity (1h)",
              "new / unseen device"]
}
```

The Streamlit dashboard shows the same reasons plus a contribution bar chart.

## 🗃️ Train on a real dataset

Beyond the synthetic data, you can train on any **numeric** fraud dataset — e.g.
Kaggle's [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(`Time`, `V1..V28`, `Amount`, `Class`):

```bash
python -m src.realdata --data creditcard.csv --target Class
```

It auto-detects the numeric feature columns, trains a balanced Random Forest and
saves a separate model to `models/fraudshield_real.joblib`.

## 📓 Exploratory data analysis

`notebooks/eda.ipynb` walks through class balance, amount/hour distributions and
feature–label correlations (run with `jupyter notebook` from the repo root).

## 🐳 Docker

Build and run the API in a container (a model is trained at build time):

```bash
docker compose up --build
# API now live at http://localhost:8000  (docs at /docs)
```

## ⚙️ Continuous Integration

Every push / PR runs the test suite plus a train+predict smoke test across
Python 3.10–3.12 via GitHub Actions (`.github/workflows/ci.yml`).

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
| POST   | `/explain`        | score **with SHAP reasons**          |
| GET    | `/stats`          | aggregate analytics (admin)          |
| GET    | `/predictions`    | recent scored transactions (admin)   |

Every scored transaction is **persisted** to the database, written to an
**audit trail**, and high-risk transactions fire **real-time alerts** (see below).

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
