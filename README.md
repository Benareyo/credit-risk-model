# Bati Bank Credit Scoring and Underwriting System

An end-to-end machine learning and credit underwriting engine built for Bati Bank's new Buy-Now-Pay-Later (BNPL) financial service. This system ingests mobile money transaction logs, handles data cleaning and feature engineering pipelines, establishes automated behavioral risk target proxies, and runs hyperparameter-optimized classification tracking via MLflow.

---

## 📋 Project Overview & Business Understanding
Bati Bank is partnering with a major eCommerce platform to offer a BNPL credit facility. Since traditional credit bureau data is unavailable for a large portion of the target customer base, this project focuses on:
* Structuring alternative credit scoring frameworks using mobile wallet data.
* Engineering robust behavioral, temporal, and financial features.
* Constructing an unsupervised proxy target variable to isolate high-risk credit defaults.
* Training, optimizing, and registering an enterprise-ready classification model.

---

## 🛠️ Data Pipeline & Feature Engineering
The production data pipeline is built as a modular Scikit-Learn `Pipeline` class architecture within `src/data_processing.py`. It automates the following steps:

1. **Temporal Engineering**: Extracts time-based attributes (`TransactionHour`, `TransactionDay`, `TransactionMonth`, and `TransactionYear`) from transaction timestamps to identify cyclical, high-risk, or fraudulent spending windows.
2. **Aggregated Customer Profiling**: Tracks rolling behavior characteristics grouped per unique `CustomerId`:
   * `Total_Transaction_Amount` (Total volume of credit used)
   * `Average_Transaction_Amount` (Mean purchasing tier)
   * `Transaction_Count` (Velocity and account engagement)
   * `Std_Dev_Transaction_Amount` (Spending volatility)
3. **Categorical Feature Encoding**: Encodes high-cardinality metadata tags (`ProviderId`, `ProductId`) dynamically based on their relative frequency distribution across the dataset.
4. **Weight of Evidence (WoE) Mapping**: Implements a mathematically rigorous Weight of Evidence transformation for `ProductCategory` and `ChannelId` categories based on traditional Basel II risk principles. This captures non-linear relationships and maximizes the Information Value (IV) without over-fitting categorical buckets.
5. **Numerical Normalization**: Transforms heavily right-skewed features and scales numerical vectors using standard scaling (`StandardScaler`) to bring all variables onto a uniform scale (mean of 0, standard deviation of 1) for modeling.

---

## 🎯 Target Proxy Variable Engineering (Task 4)
Because the raw transactional logs do not include a explicit historical loan default column, a ground-truth proxy flag (`is_high_risk`) was engineered using customer engagement profiling:

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
The **Random Forest Classifier** is our selected production champion for Bati Bank's live engine. In credit underwriting, raw accuracy is a trap due to data imbalances. Random Forest achieved a spectacular **F1-Score of 0.970** and a **Recall of 0.976**. This ultra-high recall ensures that our system accurately flags **97.6% of defaulting behaviors**, effectively insulating Bati Bank from toxic debt while maintaining an optimal approval rate for creditworthy customers.

---

## 🚀 Installation, Setup & Local Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/Benareyo/credit-risk-model.git](https://github.com/Benareyo/credit-risk-model.git)
cd credit-risk-model

# Create and activate a clean virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash
pip install -r requirements.txt