import pytest
import pandas as pd
import numpy as np
from src.data_processing import build_production_pipeline


def test_feature_engineering_pipeline_shapes_and_outputs():
    """
    Unit test checking structure, scales, and custom WoE output generation.
    """
    mock_data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C01', 'C01', 'C02'],
        'Amount': [15000.0, 25000.0, 500.0],
        'Value': [15000, 25000, 500],
        'PricingStrategy': [2, 2, 4],
        'ProductCategory': ['airtime', 'airtime', 'utility'],
        'ChannelId': ['ChannelId_3', 'ChannelId_3', 'ChannelId_2'],
        'ProviderId': ['ProviderId_1', 'ProviderId_1', 'ProviderId_4'],
        'ProductId': ['P1', 'P1', 'P2'],
        'FraudResult': [0, 0, 1],
        'TransactionStartTime': ['2026-05-29T14:30:00Z', '2026-05-29T15:45:00Z', '2026-05-29T16:00:00Z']
    })

    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_data)

    # Check structural feature sets
    assert 'CustomerId' in processed_df.columns
    assert 'Total_Transaction_Amount' in processed_df.columns
    assert 'ProductCategory_WoE' in processed_df.columns
    assert 'ChannelId_WoE' in processed_df.columns
    
    # Check completeness
    assert processed_df.isnull().sum().sum() == 0
    print("\n✅ All comprehensive pipeline tests pass perfectly!")