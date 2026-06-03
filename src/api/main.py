import os
import pathlib
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditScoringRequest

app = FastAPI(
    title="Bati Bank Credit Scoring System API",
    description="Production endpoint for serving real-time BNPL credit risk assessments under Basel II guidelines.",
    version="1.0.0"
)

# Navigate from src/api/ up to the root project boundary
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Attempting to dynamically resolve production model artifacts...")
    exp_dir = os.path.join(BASE_DIR, "mlruns", "1")
    
    # 1. Identify the unique run hash folder dynamically (skipping metadata files AND the 'models' folder)
    run_folders = [
        d for d in os.listdir(exp_dir) 
        if os.path.isdir(os.path.join(exp_dir, d)) and d not in ["meta.yaml", "models"]
    ]
    
    if not run_folders:
        raise FileNotFoundError("Could not find an active MLflow run ID folder inside 'mlruns/1/'.")
        
    latest_run = run_folders[0]
    artifacts_dir = os.path.join(exp_dir, latest_run, "artifacts")
    print(f"Targeting active Run ID directory: {latest_run}")
    
    # 2. Dynamically scan the artifacts folder to find whatever model directory is actually inside it
    model_folder_name = [f for f in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, f))][0]
    raw_path = os.path.join(artifacts_dir, model_folder_name)
    
    # 3. Convert standard Windows file path string into a robust file:// URI format
    model_uri = pathlib.Path(os.path.abspath(raw_path)).as_uri()
    print(f"Detected internal model folder: '{model_folder_name}'")
    
    model = mlflow.sklearn.load_model(model_uri)
    print(f"🚀 Success! Production model loaded from URI: {model_uri}")
except Exception as e:
    raise RuntimeError(f"Critical Failure: Unable to locate model binaries. Please verify 'mlruns/' folder exists. Details: {e}")

@app.get("/")
def health_check():
    return {
        "status": "Healthy",
        "service": "Bati Bank Credit Underwriting API",
        "winning_model_architecture": "Random Forest Classifier"
    }

@app.post("/predict")
def predict_credit_risk(payload: CreditScoringRequest):
    try:
        # Convert incoming JSON payload schema data directly back into a structured pandas DataFrame row
        input_data = pd.DataFrame([payload.dict()])
        
        # Calculate real-time binary inferences and exact default probabilities
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        
        # Structure business operational output mapping labels
        underwriting_decision = "High Risk (Reject Credit Request)" if prediction == 1 else "Low Risk (Approve Loan)"
        
        return {
            "is_high_risk_prediction": prediction,
            "default_probability_score": round(probability, 4),
            "underwriting_decision": underwriting_decision
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API Processing Error during inference calculation: {str(e)}")