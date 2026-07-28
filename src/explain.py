"""
SHAP Explainability for Bati Bank Credit Scoring Model
========================================================
Directly addresses the interpretability requirement the project's own README
argues for under Basel II: "credit models cannot function as uninterpretable
black boxes." This script trains the champion Random Forest model and
generates global + local SHAP explanations.

Run from the repo root:
    python -m src.explain
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.train import load_and_split_credit_data

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10


@dataclass
class ExplainabilityConfig:
    processed_data_path: str
    figures_dir: str
    example_customer_index: int = 0
    top_n_features: int = 10


def train_champion_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def generate_global_importance_plot(X_test, shap_values, config: ExplainabilityConfig) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=config.top_n_features)
    plt.title("Global Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, "shap_global_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def generate_beeswarm_plot(shap_values, X_test, config: ExplainabilityConfig) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=config.top_n_features)
    plt.title("Feature Effects on Default Risk Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()


def generate_local_explanation(explainer, X_test, shap_values, model, config: ExplainabilityConfig) -> dict:
    idx = config.example_customer_index
    customer_row = X_test.iloc[[idx]]
    prediction = int(model.predict(customer_row)[0])
    probability = float(model.predict_proba(customer_row)[0][1])

    expected_value = explainer.expected_value
    base_value = float(expected_value[1]) if hasattr(expected_value, "__len__") else float(expected_value)

    row_shap = shap_values[idx]
    if row_shap.ndim > 1:
        row_shap = row_shap[:, 1]

    plt.figure()
    single_shap = shap.Explanation(
        values=row_shap, base_values=base_value,
        data=customer_row.iloc[0].values, feature_names=list(X_test.columns),
    )
    shap.plots.waterfall(single_shap, show=False, max_display=config.top_n_features)
    plt.title(f"Why This Customer Was Flagged {'High-Risk' if prediction == 1 else 'Low-Risk'}")
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, "shap_local_explanation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {"customer_index": idx, "prediction": prediction, "default_probability": round(probability, 4)}


def check_concerning_patterns(shap_values, X_test, config: ExplainabilityConfig) -> list[str]:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=-1)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)

    warnings = []
    top_feature = importance.index[0]
    top_share = importance.iloc[0] / importance.sum()
    if top_share > 0.5:
        warnings.append(
            f"'{top_feature}' alone accounts for {top_share:.0%} of total feature "
            f"importance -- worth auditing for leakage from the synthetic proxy target."
        )
    if "Transaction_Count" in importance.index[:2] or "Total_Transaction_Amount" in importance.index[:2]:
        warnings.append(
            "Engagement-volume features rank at the top -- expected, since the proxy "
            "target is built from RFM/engagement clustering. Confirms the model learns "
            "the proxy as designed, but means it needs validation against real repayment "
            "outcomes before being treated as true default risk."
        )
    return warnings


def run_explainability_pipeline(config: ExplainabilityConfig) -> dict:
    os.makedirs(config.figures_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split_credit_data(config.processed_data_path)
    model = train_champion_model(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    generate_global_importance_plot(X_test, shap_values, config)
    generate_beeswarm_plot(shap_values, X_test, config)
    local_result = generate_local_explanation(explainer, X_test, shap_values, model, config)
    warnings = check_concerning_patterns(shap_values, X_test, config)

    print("SHAP explainability complete.")
    print(f"  Global importance plot:  {config.figures_dir}/shap_global_importance.png")
    print(f"  Beeswarm plot:           {config.figures_dir}/shap_beeswarm.png")
    print(f"  Local explanation plot:  {config.figures_dir}/shap_local_explanation.png")
    print(f"  Example customer #{local_result['customer_index']}: "
          f"prediction={'HIGH RISK' if local_result['prediction'] == 1 else 'LOW RISK'}, "
          f"default_probability={local_result['default_probability']}")
    if warnings:
        print("\n  Concerning patterns flagged for review:")
        for w in warnings:
            print(f"   - {w}")

    return {"local_result": local_result, "warnings": warnings}


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = ExplainabilityConfig(
        processed_data_path=os.path.join(base_dir, "data", "processed", "processed_credit_data.csv"),
        figures_dir=os.path.join(base_dir, "reports", "figures"),
    )
    run_explainability_pipeline(cfg)
