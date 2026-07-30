from pydantic import BaseModel, Field

class CreditScoringRequest(BaseModel):
    Amount: float = Field(..., description="Transaction funding amount vector element")
    Value: float = Field(..., description="Absolute transaction value element")
    PricingStrategy: float = Field(..., description="Pricing strategy class identifier")
    TransactionHour: float = Field(..., description="Hour component extracted from timestamp")
    TransactionDay: float = Field(..., description="Day component extracted from timestamp")
    TransactionMonth: float = Field(..., description="Month component extracted from timestamp")
    TransactionYear: float = Field(..., description="Year component extracted from timestamp")
    Total_Transaction_Amount: float = Field(..., description="Historic rolling aggregate transaction volume")
    Average_Transaction_Amount: float = Field(..., description="Mean historic spending tier metric")
    Transaction_Count: float = Field(..., description="Total transactional history velocity")
    Std_Dev_Transaction_Amount: float = Field(..., description="Historical spending volatility score")
    ProviderId_Encoded: float = Field(..., description="Frequency-encoded Provider identifier link")
    ProductId_Encoded: float = Field(..., description="Frequency-encoded Product identifier link")
    ProductCategory_WoE: float = Field(..., description="Weight of Evidence mapped risk transformation for Category")
    ChannelId_WoE: float = Field(..., description="Weight of Evidence mapped risk transformation for Channel")

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": -1000.0,
                "Value": 1000.0,
                "PricingStrategy": 2.0,
                "TransactionHour": 14.0,
                "TransactionDay": 29.0,
                "TransactionMonth": 5.0,
                "TransactionYear": 2026.0,
                "Total_Transaction_Amount": 25000.0,
                "Average_Transaction_Amount": 2500.0,
                "Transaction_Count": 10.0,
                "Std_Dev_Transaction_Amount": 450.0,
                "ProviderId_Encoded": 0.154,
                "ProductId_Encoded": 0.085,
                "ProductCategory_WoE": 0.231,
                "ChannelId_WoE": -0.114
            }
        }



        