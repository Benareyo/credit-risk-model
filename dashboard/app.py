"""
Bati Bank Credit Scoring Dashboard
====================================
Interactive exploration of model performance, individual predictions with
SHAP explainability, and illustrative business impact -- built for a finance
sector audience per the Week 12 capstone brief.

Run from the repo root (after generating data/processed/processed_credit_data.csv
via `python -m src.data_processing`):
    streamlit run dashboard/app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)

from src.train import load_and_split_credit_data

st.set_page_config(page_title="Bati Bank Credit Scoring Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_credit_data.csv")

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10


# ---------------------------------------------------------------------------
# Cached data + model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = load_and_split_credit_data(DATA_PATH)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    return rf, lr, X_train, X_test, y_train, y_test


@st.cache_resource
def build_shap_explainer(_model, X_train_sample: pd.DataFrame):
    return shap.TreeExplainer(_model)


def compute_metrics(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1 Score": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }


df = load_data()
rf_model, lr_model, X_train, X_test, y_train, y_test = train_models()
explainer = build_shap_explainer(rf_model, X_train)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Bati Bank Credit Scoring")
page = st.sidebar.radio("Navigate", ["Overview", "Explore & Predict", "Explainability", "Business Impact"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Champion model: Random Forest (per src/train.py's own comparison: "
    "F1=0.970, ROC-AUC=1.000 vs. Logistic Regression's 0.720/0.995)."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Known issue: the feature engineering pipeline's scaler currently "
    "re-fits on every `transform()` call rather than reusing the fit from "
    "training -- safe for batch scoring on the full dataset (as used here), "
    "but not yet safe for single new-row real-time inference. Flagged for "
    "the Monday engineering pass."
)

# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if page == "Overview":
    st.title("Overview")
    st.caption("Key metrics and dataset summary for Bati Bank's BNPL credit risk model")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Scored", f"{len(df):,}")
    col2.metric("Flagged High-Risk", f"{(df['is_high_risk'] == 1).sum():,}",
                f"{(df['is_high_risk'] == 1).mean():.1%} of portfolio")
    col3.metric("Flagged Low-Risk", f"{(df['is_high_risk'] == 0).sum():,}")

    rf_metrics = compute_metrics(rf_model, X_test, y_test)
    col4.metric("Champion Model F1 Score", f"{rf_metrics['F1 Score']:.3f}")

    st.markdown("---")
    st.subheader("Model Comparison")
    lr_metrics = compute_metrics(lr_model, X_test, y_test)
    comparison_df = pd.DataFrame({
        "Logistic Regression": lr_metrics,
        "Random Forest (champion)": rf_metrics,
    }).T
    st.dataframe(comparison_df.style.format("{:.3f}"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk Distribution")
        risk_counts = df["is_high_risk"].map({0: "Low Risk", 1: "High Risk"}).value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(risk_counts.index, risk_counts.values, color=["#4C72B0", "#C44E52"])
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Confusion Matrix (Champion Model, Test Set)")
        preds = rf_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Low", "Predicted High"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Low", "Actual High"])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Regulatory Context (Basel II)")
    st.write(
        "Credit models cannot function as uninterpretable black boxes under Basel II -- "
        "every approval, rejection, or pricing decision needs a clear, auditable explanation. "
        "The **Explainability** page addresses this directly with global and per-customer SHAP "
        "explanations for the champion Random Forest model."
    )

# ===========================================================================
# PAGE: EXPLORE & PREDICT
# ===========================================================================
elif page == "Explore & Predict":
    st.title("Explore & Predict")
    st.caption("Select a customer to view their engineered risk profile and the model's live prediction")

    test_indices = X_test.index.tolist()
    selected_idx = st.selectbox(
        "Select a customer (by row index in the held-out test set)",
        options=test_indices,
        format_func=lambda i: f"Test customer #{i}"
    )

    customer_row = X_test.loc[[selected_idx]]
    actual_label = y_test.loc[selected_idx]

    prediction = int(rf_model.predict(customer_row)[0])
    probability = float(rf_model.predict_proba(customer_row)[0][1])
    decision = "High Risk (Reject Credit Request)" if prediction == 1 else "Low Risk (Approve Loan)"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Prediction", "HIGH RISK" if prediction == 1 else "LOW RISK")
    col2.metric("Default Probability", f"{probability:.1%}")
    col3.metric("Actual Label (held out)", "HIGH RISK" if actual_label == 1 else "LOW RISK")

    if prediction == actual_label:
        st.success(f"Underwriting Decision: {decision} -- matches held-out actual label")
    else:
        st.warning(f"Underwriting Decision: {decision} -- does NOT match held-out actual label "
                   f"(model disagreement, useful for reviewing edge cases)")

    st.markdown("---")
    st.subheader("Engineered Feature Values for This Customer")
    st.dataframe(customer_row.T.rename(columns={selected_idx: "Value"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Why the Model Made This Decision")
    row_position = X_test.index.get_loc(selected_idx)
    shap_values_raw = explainer.shap_values(customer_row)
    if isinstance(shap_values_raw, list):
        row_shap = shap_values_raw[1][0]
    elif shap_values_raw.ndim == 3:
        row_shap = shap_values_raw[0, :, 1]
    else:
        row_shap = shap_values_raw[0]

    expected_value = explainer.expected_value
    base_value = float(expected_value[1]) if hasattr(expected_value, "__len__") else float(expected_value)

    fig, ax = plt.subplots(figsize=(9, 5))
    single_shap = shap.Explanation(
        values=row_shap, base_values=base_value,
        data=customer_row.iloc[0].values, feature_names=list(X_test.columns),
    )
    shap.plots.waterfall(single_shap, show=False, max_display=10)
    st.pyplot(fig)
    plt.close(fig)

# ===========================================================================
# PAGE: EXPLAINABILITY
# ===========================================================================
elif page == "Explainability":
    st.title("Model Explainability")
    st.caption("Global feature importance and effect direction, computed live via SHAP TreeExplainer")

    sample = X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE)
    shap_values_raw = explainer.shap_values(sample)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=10)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Feature Effects (Direction + Magnitude)")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(shap_values, sample, show=False, max_display=10)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Concerning Patterns")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=sample.columns).sort_values(ascending=False)
    top_share = importance.iloc[0] / importance.sum()

    if top_share > 0.5:
        st.warning(
            f"**{importance.index[0]}** alone accounts for **{top_share:.0%}** of total feature "
            f"importance. A single feature this dominant is worth auditing for leakage from the "
            f"synthetic proxy target (`is_high_risk` is derived from RFM/engagement clustering, "
            f"not a real historical default)."
        )
    else:
        st.info(f"No single feature dominates (top feature: {importance.index[0]}, {top_share:.0%} share).")

    st.write(
        "**Interpretation note:** the top-ranked features are consistently engagement-volume "
        "metrics (Transaction_Count, Total_Transaction_Amount). This is expected, since the "
        "`is_high_risk` proxy target is itself built from RFM/engagement clustering rather than "
        "observed loan defaults -- the model is correctly learning the proxy it was given, but "
        "this should be validated against real repayment outcomes before being treated as a "
        "true default-risk signal in production."
    )

# ===========================================================================
# PAGE: BUSINESS IMPACT
# ===========================================================================
elif page == "Business Impact":
    st.title("Business Impact")
    st.caption("Illustrative translation of model performance into business terms -- adjust the assumptions below")

    st.info(
        "These figures are **illustrative**, driven by the sliders below, not measured outcomes. "
        "Bati Bank has no historical default-loss data in this dataset to calibrate against; this "
        "calculator exists to make the precision/recall tradeoff concrete for non-technical "
        "stakeholders, not to claim a validated dollar figure."
    )

    col1, col2 = st.columns(2)
    avg_loan_size = col1.slider("Assumed average loan size ($)", 50, 2000, 300, step=25)
    default_loss_rate = col2.slider("Assumed loss-given-default (% of loan)", 10, 100, 70, step=5) / 100

    threshold = st.slider("Decision threshold (probability above which a loan is rejected)",
                           0.05, 0.95, 0.50, step=0.05)

    probs = rf_model.predict_proba(X_test)[:, 1]
    preds_at_threshold = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds_at_threshold).ravel()

    avoided_losses = tp * avg_loan_size * default_loss_rate
    foregone_revenue = fp * avg_loan_size * 0.15  # illustrative margin assumption on wrongly-rejected good loans
    missed_losses = fn * avg_loan_size * default_loss_rate

    col1, col2, col3 = st.columns(3)
    col1.metric("True Positives (correctly rejected defaulters)", tp,
                f"~${avoided_losses:,.0f} losses avoided")
    col2.metric("False Positives (good customers wrongly rejected)", fp,
                f"~${foregone_revenue:,.0f} revenue foregone")
    col3.metric("False Negatives (defaulters wrongly approved)", fn,
                f"~${missed_losses:,.0f} losses not caught")

    st.markdown("---")
    st.subheader("Threshold Tradeoff")
    thresholds = np.linspace(0.05, 0.95, 19)
    tradeoff_rows = []
    for t in thresholds:
        p = (probs >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, p).ravel()
        net_value = (tp_ * avg_loan_size * default_loss_rate) - (fp_ * avg_loan_size * 0.15) - (fn_ * avg_loan_size * default_loss_rate)
        tradeoff_rows.append({"threshold": round(t, 2), "net_illustrative_value": net_value})
    tradeoff_df = pd.DataFrame(tradeoff_rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tradeoff_df["threshold"], tradeoff_df["net_illustrative_value"], marker="o", color="#4C72B0")
    ax.axvline(threshold, color="#C44E52", linestyle="--", label="Current threshold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Net Illustrative Value ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This chart shows how the illustrative net value changes as the rejection threshold moves. "
        "A lower threshold rejects more borderline customers (fewer missed defaults, more foregone "
        "revenue); a higher threshold approves more of them (opposite tradeoff)."
    )
