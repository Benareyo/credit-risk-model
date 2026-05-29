import pytest
import pandas as pd
import numpy as np
from src.data_processing import build_production_pipeline


def test_feature_engineering_pipeline_shapes_and_outputs():
    """
    Unit test to confirm that our Scikit-Learn data processing pipeline 
    successfully creates engineered features, extracts date arrays, and handles scaling.
    """
    # Create fake transactional sample records
    mock_data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C01', 'C01', 'C02'],
        'Amount': [15000.0, 25000.0, 500.0],
        'Value': [15000, 25000, 500],
        'PricingStrategy': [2, 2, 4],
        'ProductCategory': ['airtime', 'airtime', 'utility'],
        'ChannelId': ['ChannelId_3', 'ChannelId_3', 'ChannelId_2'],
        'ProviderId': ['ProviderId_1', 'ProviderId_1', 'ProviderId_4'],
        'TransactionStartTime': ['2026-05-29T14:30:00Z', '2026-05-29T15:45:00Z', '2026-05-29T16:00:00Z']
    })

    # Execute pipeline engine
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_data)

    # Assert basic structure and column properties
    assert 'CustomerId' in processed_df.columns
    assert 'Total_Transaction_Amount' in processed_df.columns
    assert 'TransactionHour' in processed_df.columns
    assert 'ProductCategory_Encoded' in processed_df.columns
    
    # Check that data contains zero null cells post-transformation
    assert processed_df.isnull().sum().sum() == 0
    print("\n✅ All custom pipeline test assertions passed completely!")