# Bati Bank Credit Risk Probability Model for Alternative Data

An end-to-end industrial credit scoring engine built to translate eCommerce behavioral transaction streams into regulatory-compliant predictive credit risk signals under the Basel II Capital Accord framework.

---

## 🏛️ Credit Scoring Business Understanding

### 1. Basel II Compliance & The Interpretability Imperative
The Basel II Capital Accord establishes stringent global regulatory requirements concerning capital adequacy, risk asset management, and operational transparency. In a regulated financial environment like Bati Bank, predictive credit models cannot function as uninterpretable "black boxes." 

When an underwriting system impacts credit approvals, loan limits, or interest pricing, the bank is legally obligated to provide a clear, auditable explanation for its risk decisions. This makes model interpretability an absolute business and legal requirement. We ensure this compliance by leveraging **Weight of Evidence (WoE)** transformations and structured feature engineering pipelines, creating transparent, auditable paths that independent risk management auditors can easily validate.

### 2. Synthetic Proxy Target Variable Design & Institutional Risk
Because the Xente eCommerce dataset consists entirely of transactional records without historical credit default labels, we must programmatically construct a **Proxy Target Variable (`is_high_risk`)** using Recency, Frequency, and Monetary (RFM) customer segmentation. Customers exhibiting low platform engagement, erratic transaction patterns, and minimal monetary value are mathematically clustered to serve as the high-risk baseline.

While analytical proxying allows us to build a predictive model without prior credit history, it introduces two distinct structural business risks to Bati Bank:
* **Type I Error (False Positive Risk):** Stable, high-net-worth customers who simply use the eCommerce platform infrequently may be misclassified as "High-Risk." This leads to credit denial, hurting customer acquisition and leaving low-risk transaction revenue on the table.
* **Type II Error (False Negative Risk):** Highly active, high-frequency users might be classified as "Low-Risk" based purely on transaction volume, failing to catch underlying insolvency. This misclassification results in bad loans, direct capital leakage, and increased default rates for Bati Bank.

### 3. Structural Trade-offs: Interpretable vs. Non-Linear Models
Deploying a credit risk asset requires balancing regulatory constraints against maximum predictive power:

* **Logistic Regression with Weight of Evidence (WoE):**
  * *Advantages:* Highly linear, transparent, and directly maps to traditional points-based credit scorecards. It allows risk officers to see the exact risk weight of every attribute, ensuring seamless Basel II approval.
  * *Disadvantages:* Cannot naturally capture complex, non-linear relationships or multi-feature interactions within behavioral data.
* **Gradient Boosting Ensembles (e.g., XGBoost, LightGBM):**
  * *Advantages:* Exceptional classification performance, inherently handles highly skewed or imbalanced financial data, and optimizes default detection accuracy.
  * *Disadvantages:* Highly complex architectures that are difficult to explain to financial regulators without implementing resource-heavy post-hoc interpretability frameworks (such as SHAP value matrices).

---

## 📂 Project Directory Architecture

```text
credit-risk-model/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI/CD Pipeline
├── data/
│   ├── raw/                       # Unprocessed Xente Transaction Data
│   └── processed/                 # Model-Ready Features with Target Variables
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis & Visualizations
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI Engine Application
│   │   └── pydantic_models.py     # API Strict Type Validation Schemas
│   ├── __init__.py
│   ├── data_processing.py         # Automated Scaling & WoE Feature Pipelines
│   ├── train.py                   # Model Training & MLflow Experimentation
│   └── predict.py                 # Batch & Real-time Inference Engine
├── tests/
│   └── test_data_processing.py    # Automated Pytest Suite
├── Dockerfile                     # Microservice Containerization Configuration
├── docker-compose.yml             # Local Orchestration Manifest
├── requirements.txt               # Locked System Dependencies
└── README.md                      # Operational Documentation