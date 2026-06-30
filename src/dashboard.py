"""Streamlit dashboard for FraudShield AI.

A browser UI to score transactions interactively and explore model behaviour.

Run with::

    streamlit run src/dashboard.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# `streamlit run src/dashboard.py` executes this file as a top-level script, so
# the `src` package isn't on the path and relative imports won't resolve. Add
# the project root and use absolute imports, which also work under `-m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import DEVICE_TYPES, MERCHANT_CATEGORIES  # noqa: E402
from src.predict import DEFAULT_MODEL_PATH, FraudDetector  # noqa: E402

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, threshold: float) -> FraudDetector | None:
    """Load (and cache) the trained model, or None if it isn't trained yet."""
    if not os.path.exists(model_path):
        return None
    return FraudDetector(model_path=model_path, threshold=threshold)


RISK_COLORS = {
    "HIGH": "#d62728",
    "MEDIUM": "#ff7f0e",
    "LOW": "#1f77b4",
    "MINIMAL": "#2ca02c",
}


def main() -> None:
    st.title("🛡️ FraudShield AI")
    st.caption("Random Forest fraud detection for bank / online transactions")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input("Model path", DEFAULT_MODEL_PATH)
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown(
            "Train a model first if none exists:\n\n```bash\npython -m src.train\n```"
        )

    detector = load_detector(model_path, threshold)
    if detector is None:
        st.error(
            f"No model found at `{model_path}`. "
            "Train one with `python -m src.train` and reload."
        )
        st.stop()

    tab_single, tab_batch, tab_admin = st.tabs(
        ["🔎 Single transaction", "📁 Batch CSV", "📊 Admin analytics"]
    )

    with tab_single:
        _single_transaction_tab(detector)

    with tab_batch:
        _batch_tab(detector)

    with tab_admin:
        _admin_tab()


def _single_transaction_tab(detector: FraudDetector) -> None:
    st.subheader("Score a single transaction")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=1450.0, step=10.0)
        hour = st.slider("Hour of day", 0, 23, 3)
        account_age_days = st.number_input("Account age (days)", min_value=0, value=12)
    with col2:
        txn_count_1h = st.number_input("Txns in last 1h", min_value=0, value=7)
        txn_count_24h = st.number_input("Txns in last 24h", min_value=0, value=25)
        merchant_category = st.selectbox("Merchant category", MERCHANT_CATEGORIES, index=MERCHANT_CATEGORIES.index("money_transfer"))
    with col3:
        device_type = st.selectbox("Device type", DEVICE_TYPES, index=DEVICE_TYPES.index("web"))
        foreign_transaction = 1 if st.checkbox("Foreign / cross-border", value=True) else 0
        is_new_device = 1 if st.checkbox("New / unseen device", value=True) else 0

    if st.button("🔍 Check for fraud", type="primary"):
        transaction = {
            "amount": amount,
            "hour": hour,
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "foreign_transaction": foreign_transaction,
            "account_age_days": account_age_days,
            "is_new_device": is_new_device,
            "merchant_category": merchant_category,
            "device_type": device_type,
        }
        result = detector.score(transaction)

        # Persist + alert so the Admin tab and any alert channel stay in sync.
        try:
            from src import alerts, db

            db.init_db()
            db.record_prediction(transaction, result, api_key_name="dashboard")
            alerts.send_alert(transaction, result)
        except Exception:
            pass

        color = RISK_COLORS.get(result["risk_level"], "#333")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud probability", f"{result['fraud_probability']:.0%}")
        c2.metric("Verdict", "FRAUD" if result["is_fraud"] else "LEGIT")
        c3.markdown(
            f"<div style='padding:0.5rem 1rem;border-radius:8px;background:{color};"
            f"color:white;text-align:center;font-weight:700;font-size:1.1rem'>"
            f"Risk: {result['risk_level']}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, result["fraud_probability"]))

        _show_explanation(transaction)


def _show_explanation(transaction: dict) -> None:
    """Render SHAP-based reasons under the verdict (best-effort)."""
    with st.expander("🔍 Why? (SHAP explanation)", expanded=True):
        try:
            from src.explain import get_explainer

            explained = get_explainer(DEFAULT_MODEL_PATH).explain(transaction)
        except Exception as exc:  # shap missing or any runtime issue
            st.info(f"Explanation unavailable: {exc}")
            return

        reasons = explained.get("reasons") or []
        if reasons:
            st.write("**Top risk factors:**")
            for reason in reasons:
                st.write(f"- {reason}")
        contributions = explained.get("explanation", [])
        if contributions:
            chart_df = pd.DataFrame(contributions).set_index("feature")["shap_value"]
            st.bar_chart(chart_df)


def _batch_tab(detector: FraudDetector) -> None:
    st.subheader("Score a batch of transactions")
    st.write(
        "Upload a CSV with the model's feature columns "
        "(`amount, hour, txn_count_1h, txn_count_24h, foreign_transaction, "
        "account_age_days, is_new_device, merchant_category, device_type`)."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    try:
        results = detector.score_many(df.to_dict(orient="records"))
    except ValueError as exc:
        st.error(str(exc))
        return

    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    fraud_count = int(scored["is_fraud"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Transactions", len(scored))
    c2.metric("Flagged as fraud", fraud_count)

    st.dataframe(scored, use_container_width=True)
    st.download_button(
        "⬇️ Download scored CSV",
        scored.to_csv(index=False).encode(),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


def _admin_tab() -> None:
    """Live analytics over everything the API/dashboard has scored."""
    st.subheader("Operations analytics")
    st.caption("Aggregated from the FraudShield database (all scored transactions).")

    from src import db

    db.init_db()
    summary = db.stats_summary()

    if summary["total_transactions"] == 0:
        st.info(
            "No transactions scored yet. Score some on the other tabs or via the "
            "REST API, then refresh."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scored", f"{summary['total_transactions']:,}")
    c2.metric("Flagged fraud", f"{summary['fraud_flagged']:,}")
    c3.metric("Fraud rate", f"{summary['fraud_rate']:.1%}")
    c4.metric("Avg fraud prob.", f"{summary['avg_fraud_probability']:.0%}")

    if summary["by_risk_level"]:
        st.write("**By risk level**")
        risk_order = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]
        risk_df = pd.DataFrame(
            {"count": [summary["by_risk_level"].get(r, 0) for r in risk_order]},
            index=risk_order,
        )
        st.bar_chart(risk_df)

    st.write("**Most recent transactions**")
    recent = db.recent_predictions(limit=50)
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True)


main()
