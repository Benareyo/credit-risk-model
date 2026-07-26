# Bati Bank Credit Scoring and Underwriting System

An end-to-end machine learning and credit underwriting engine built to translate eCommerce behavioral transaction streams into regulatory-compliant predictive credit risk signals for Bati Bank's new Buy-Now-Pay-Later (BNPL) financial service. This system ingests mobile money transaction logs, handles automated feature engineering pipelines, establishes behavioral risk target proxies, and runs hyperparameter-optimized classification tracking via MLflow.

---

## 🏛️ Credit Scoring Business Understanding

### 1. Basel II Compliance & The Interpretability Imperative
The Basel II Capital Accord establishes stringent global regulatory requirements concerning capital adequacy, risk asset management, and operational transparency. In a regulated financial environment like Bati Bank, predictive credit models cannot function as uninterpretable "black boxes." 

When an underwriting system impacts credit approvals, loan limits, or interest pricing, the bank is legally obligated to provide a clear, auditable explanation for its risk decisions. This makes model interpretability an absolute business and legal requirement. We ensure this compliance by leveraging Weight of Evidence (WoE) transformations and structured feature engineering pipelines, creating transparent, auditable paths that independent risk management auditors can easily validate.

### 2. Synthetic Proxy Target Variable Design & Institutional Risk
Because the raw transactional logs do not include an explicit historical loan default column, a ground-truth proxy flag (`is_high_risk`) was engineered using customer engagement profiling (Recency, Frequency, and Monetary metrics). While analytical proxying allows us to build a predictive model without prior credit history, it introduces two distinct structural business risks to Bati Bank:
* **Type I Error (False Positive Risk):** Stable, high-net-worth customers who simply use the eCommerce platform infrequently may be misclassified as "High-Risk." This leads to unnecessary credit denial, hurting customer acquisition and leaving low-risk transaction revenue on the table.
* **Type II Error (False Negative Risk):** Highly active, high-frequency users might be classified as "Low-Risk" based purely on transaction volume, failing to catch underlying insolvency. This misclassification results in bad loans, direct capital leakage, and increased default rates for Bati Bank.

### 3. Structural Trade-offs: Interpretable vs. Non-Linear Models
Deploying a credit risk asset requires balancing regulatory constraints against maximum predictive power:
* **Logistic Regression with Weight of Evidence (WoE):** Highly linear, transparent, and directly maps to traditional points-based credit scorecards. It allows risk officers to see the exact risk weight of every attribute, ensuring seamless Basel II approval, but it cannot naturally capture complex, non-linear feature interactions.
* **Random Forest / Ensemble Classifiers:** Exceptional classification performance, inherently handles highly skewed or imbalanced financial data, and optimizes default detection accuracy. However, they are architecturally more complex and require model interpretability frameworks to explain to regulators.

---

## 🛠️ Data Pipeline & Feature Engineering
The production data pipeline is built as a modular Scikit-Learn `Pipeline` class architecture within `src/data_processing.py`. It automates the following steps:

1. **Temporal Engineering**: Extracts time-based attributes (`TransactionHour`, `TransactionDay`, `TransactionMonth`, and `TransactionYear`) from transaction timestamps to identify cyclical spending windows.
2. **Aggregated Customer Profiling**: Tracks rolling behavior characteristics grouped per unique `CustomerId`:
   * `Total_Transaction_Amount` (Total volume of credit used)
   * `Average_Transaction_Amount` (Mean purchasing tier)
   * `Transaction_Count` (Velocity and account engagement)
   * `Std_Dev_Transaction_Amount` (Spending volatility)
3. **Categorical Feature Encoding**: Encodes high-cardinality metadata tags (`ProviderId`, `ProductId`) dynamically based on their relative frequency distribution across the dataset.
4. **Weight of Evidence (WoE) Mapping**: Implements a mathematically rigorous Weight of Evidence transformation for `ProductCategory` and `ChannelId` categories based on traditional Basel II risk principles, maximizing Information Value (IV) without over-fitting categorical buckets.
5. **Numerical Normalization**: Transforms heavily right-skewed features and scales numerical vectors using standard scaling (`StandardScaler`) to bring all variables onto a uniform scale (mean of 0, standard deviation of 1) for modeling.

---

## 🎯 Target Proxy Variable Engineering (Task 4)
* **RFM Aggregation**: Computes Recency (days since the customer's last transaction relative to a global dataset snapshot date), Frequency (total transaction velocity), and Monetary (absolute volume of funds moved) metrics per customer.
* **K-Means Clustering**: Transforms skewed RFM metrics via `np.log1p` and uses a standard K-Means algorithm ($K=3$, `random_state=42`) to segment the customer base into 3 distinct behavioral groups.
* **Risk Mapping**: Evaluates cluster centers to programmatically isolate the least engaged segment (Cluster 0, characterized by low transactional volume and frequency). Customers matching this profile are assigned a credit-risk target flag of `1` (High Default Risk), while all other stable customers are assigned `0`.

---

## 📊 Model Training & MLflow Experiment Tracking (Task 5)
Using our engineered features and the unsupervised `is_high_risk` target label, we developed an optimized hyperparameter grid search training loop inside `src/train.py`, tracked natively with **MLflow**.

### Performance Evaluation Matrix
The models were evaluated side-by-side using the interactive MLflow run comparison dashboard:

| Model Architecture | Hyperparameters Evaluated | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C: [0.1, 1.0, 10.0]` | 0.987 | 0.897 | 0.601 | 0.720 | 0.995 |
| **Random Forest** | `n_estimators: [50, 100]`, `max_depth: [5, 10]` | **0.998** | **0.965** | **0.976** | **0.970** | **1.000** |

### Operational Recommendation & Underwriting Justification
The **Random Forest Classifier** is our selected production champion for Bati Bank's live engine. In credit underwriting, raw accuracy is a trap due to data imbalances. Random Forest achieved a spectacular **F1-Score of 0.970** and a **Recall of 0.976**. This ultra-high recall ensures that our system accurately flags **97.6% of defaulting behaviors**, effectively insulating Bati Bank from toxic debt and reducing Type II error risks while maintaining an optimal approval rate for creditworthy customers.

---

## 📂 Project Directory Architecture

```text
credit-risk-model/
├── .github/workflows/ci.yml        # Automated GitHub Actions CI/CD Pipeline
├── data/
│   ├── raw/                        # Unprocessed Xente Transaction Data
│   └── processed/                  # Model-Ready Features with Target Variables
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis & Visualizations
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI Engine Application
│   │   └── pydantic_models.py      # API Strict Type Validation Schemas
│   ├── __init__.py
│   ├── data_processing.py          # Automated Scaling & WoE Feature Pipelines
│   ├── train.py                    # Model Training & MLflow Experimentation
│   └── predict.py                  # Batch & Real-time Inference Engine
├── tests/
│   └── test_data_processing.py     # Automated Pytest Suite
├── requirements.txt                # Locked System Dependencies
└── README.md                       # Operational Documentation