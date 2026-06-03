import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_split_credit_data(processed_data_path, random_state=42):
    """Loads engineered datasets and prepares reproducible splits."""
    df = pd.read_csv(processed_data_path)
    
    # Drop identifying structural keys to focus purely on modeling variables
    X = df.drop(columns=['CustomerId', 'is_high_risk'], errors='ignore')
    y = df['is_high_risk']
    
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

def train_and_track_models():
    """Trains Logistic Regression and Random Forest models with tuning and MLflow logging."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'processed_credit_data.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Processed training file not found at: {data_path}")
        return
        
    X_train, X_test, y_train, y_test = load_and_split_credit_data(data_path)
    
    # Establish MLflow local tracking experiment workspace
    mlflow.set_experiment("BatiBank_Credit_Scoring_System")
    
    # ------------------ MODEL 1: LOGISTIC REGRESSION ------------------
    with mlflow.start_run(run_name="Logistic_Regression_GridSearch"):
        print("\n🚀 Commencing Logistic Regression Training and Tuning...")
        lr_base = LogisticRegression(max_iter=1000, random_state=42)
        lr_param_grid = {'C': [0.1, 1.0, 10.0]}
        
        lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=3, scoring='f1', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        best_lr = lr_grid.best_estimator_
        
        # Infer predictions and calculate diagnostic metrics
        preds = best_lr.predict(X_test)
        probs = best_lr.predict_proba(X_test)[:, 1]
        
        # Log Hyperparameters
        mlflow.log_params(lr_grid.best_params_)
        mlflow.log_param("model_type", "Logistic_Regression")
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))
        
        # Log Model Binary
        mlflow.sklearn.log_model(best_lr, "logistic_regression_model")
        print("✅ Logistic Regression logged successfully to MLflow.")

    # ------------------ MODEL 2: RANDOM FOREST ------------------
    with mlflow.start_run(run_name="Random_Forest_GridSearch"):
        print("\n🚀 Commencing Random Forest Training and Tuning...")
        rf_base = RandomForestClassifier(random_state=42)
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        
        rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        
        # Infer predictions
        preds_rf = best_rf.predict(X_test)
        probs_rf = best_rf.predict_proba(X_test)[:, 1]
        
        # Log Hyperparameters
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_param("model_type", "Random_Forest")
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds_rf))
        mlflow.log_metric("precision", precision_score(y_test, preds_rf))
        mlflow.log_metric("recall", recall_score(y_test, preds_rf))
        mlflow.log_metric("f1_score", f1_score(y_test, preds_rf))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs_rf))
        
        # Log Model Binary and Register as Best Model Candidate
        mlflow.sklearn.log_model(best_rf, "random_forest_model", registered_model_name="BatiBank_Best_Credit_Model")
        print("✅ Random Forest logged and registered successfully in MLflow Model Registry.")

if __name__ == "__main__":
    train_and_track_models()