import os
import pathlib
from contextlib import asynccontextmanager
from typing import Optional

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CreditScoringRequest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_model = None
_model_load_error: Optional[str] = None


def _resolve_latest_model_uri() -> str:
    exp_dir = os.path.join(BASE_DIR, "mlruns", "1")
    run_folders = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d)) and d not in ["meta.yaml", "models"]
    ]
    if not run_folders:
        raise FileNotFoundError("Could not find an active MLflow run ID folder inside 'mlruns/1/'.")

    latest_run = run_folders[0]
    artifacts_dir = os.path.join(exp_dir, latest_run, "artifacts")
    model_folder_name = [f for f in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, f))][0]
    raw_path = os.path.join(artifacts_dir, model_folder_name)
    return pathlib.Path(os.path.abspath(raw_path)).as_uri()


def load_model() -> None:
    global _model, _model_load_error
    try:
        model_uri = _resolve_latest_model_uri()
        _model = mlflow.sklearn.load_model(model_uri)
        _model_load_error = None
        print(f"Success: production model loaded from URI: {model_uri}")
    except Exception as e:
        _model_load_error = str(e)
        print(f"Warning: model not loaded at startup. Details: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Bati Bank Credit Scoring System API",
    description="Production endpoint for serving real-time BNPL credit risk assessments under Basel II guidelines.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    return {
        "status": "Healthy" if _model is not None else "Degraded (model not loaded)",
        "service": "Bati Bank Credit Underwriting API",
        "winning_model_architecture": "Random Forest Classifier",
        "model_load_error": _model_load_error,
    }


@app.post("/predict")
def predict_credit_risk(payload: CreditScoringRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {_model_load_error or 'not yet loaded'}"
        )
    try:
        input_data = pd.DataFrame([payload.dict()])
        prediction = int(_model.predict(input_data)[0])
        probability = float(_model.predict_proba(input_data)[0][1])
        underwriting_decision = "High Risk (Reject Credit Request)" if prediction == 1 else "Low Risk (Approve Loan)"
        return {
            "is_high_risk_prediction": prediction,
            "default_probability_score": round(probability, 4),
            "underwriting_decision": underwriting_decision
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API Processing Error during inference calculation: {str(e)}")
