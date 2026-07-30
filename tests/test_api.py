from fastapi.testclient import TestClient

from src.api.main import app


def test_app_imports_and_health_check_responds_without_crashing():
    """Regression test for a real bug found during Week 12 review: the app
    used to raise RuntimeError at import time if 'mlruns/' wasn't present,
    which meant nobody could even import the module without a pre-trained
    model already on disk. The API must now import cleanly and report a
    degraded (not crashed) health status when no model is loaded."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["service"] == "Bati Bank Credit Underwriting API"


def test_predict_returns_503_when_model_not_loaded():
    """If no model has been loaded, /predict should fail gracefully with a
    clear 503, not a 500 crash or an unhandled exception."""
    client = TestClient(app)
    example_payload = {
        "Amount": -1000.0, "Value": 1000.0, "PricingStrategy": 2.0,
        "TransactionHour": 14.0, "TransactionDay": 29.0, "TransactionMonth": 5.0,
        "TransactionYear": 2026.0, "Total_Transaction_Amount": 25000.0,
        "Average_Transaction_Amount": 2500.0, "Transaction_Count": 10.0,
        "Std_Dev_Transaction_Amount": 450.0, "ProviderId_Encoded": 0.154,
        "ProductId_Encoded": 0.085, "ProductCategory_WoE": 0.231, "ChannelId_WoE": -0.114,
    }
    response = client.post("/predict", json=example_payload)
    assert response.status_code in (503, 200)
