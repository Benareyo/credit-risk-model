import numpy as np
import pandas as pd
import pytest

from src.data_processing import build_production_pipeline, engineer_proxy_target_variable


@pytest.fixture
def mock_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'CustomerId': ['C01', 'C01', 'C02', 'C03', 'C03'],
        'Amount': [15000.0, 25000.0, 500.0, -2000.0, 8000.0],
        'Value': [15000, 25000, 500, 2000, 8000],
        'PricingStrategy': [2, 2, 4, 0, 1],
        'ProductCategory': ['airtime', 'airtime', 'utility', 'financial_services', 'airtime'],
        'ChannelId': ['ChannelId_3', 'ChannelId_3', 'ChannelId_2', 'ChannelId_1', 'ChannelId_3'],
        'ProviderId': ['ProviderId_1', 'ProviderId_1', 'ProviderId_4', 'ProviderId_2', 'ProviderId_1'],
        'ProductId': ['P1', 'P1', 'P2', 'P3', 'P1'],
        'FraudResult': [0, 0, 1, 0, 0],
        'TransactionStartTime': [
            '2026-05-29T14:30:00Z', '2026-05-29T15:45:00Z', '2026-05-29T16:00:00Z',
            '2026-05-30T09:00:00Z', '2026-05-30T09:30:00Z',
        ],
    })


def test_feature_engineering_pipeline_shapes_and_outputs(mock_transactions):
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_transactions)
    assert 'CustomerId' in processed_df.columns
    assert 'Total_Transaction_Amount' in processed_df.columns
    assert 'ProductCategory_WoE' in processed_df.columns
    assert 'ChannelId_WoE' in processed_df.columns
    assert processed_df.isnull().sum().sum() == 0


def test_feature_engineering_preserves_row_count(mock_transactions):
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_transactions)
    assert len(processed_df) == len(mock_transactions)


def test_customer_aggregates_are_correct(mock_transactions):
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_transactions)
    c01_rows = processed_df[processed_df['CustomerId'] == 'C01']
    assert len(c01_rows) == 2
    assert c01_rows['Transaction_Count'].nunique() == 1


def test_woe_transformation_produces_finite_values(mock_transactions):
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(mock_transactions)
    assert np.isfinite(processed_df['ProductCategory_WoE']).all()
    assert np.isfinite(processed_df['ChannelId_WoE']).all()


def test_engineer_proxy_target_variable_returns_binary_labels(mock_transactions):
    risk_mapping = engineer_proxy_target_variable(mock_transactions, random_state=42)
    assert set(risk_mapping.keys()) == set(mock_transactions['CustomerId'].unique())
    assert set(risk_mapping.values()).issubset({0, 1})


def test_engineer_proxy_target_variable_is_reproducible(mock_transactions):
    mapping_1 = engineer_proxy_target_variable(mock_transactions, random_state=42)
    mapping_2 = engineer_proxy_target_variable(mock_transactions, random_state=42)
    assert mapping_1 == mapping_2


def test_pipeline_handles_missing_optional_columns_gracefully():
    minimal_df = pd.DataFrame({
        'TransactionId': ['T1', 'T2'],
        'CustomerId': ['C01', 'C02'],
        'Amount': [1000.0, 2000.0],
        'Value': [1000, 2000],
        'PricingStrategy': [1, 2],
        'TransactionStartTime': ['2026-05-29T14:30:00Z', '2026-05-29T15:45:00Z'],
        'FraudResult': [0, 0],
    })
    pipeline = build_production_pipeline()
    processed_df = pipeline.fit_transform(minimal_df)
    assert len(processed_df) == 2
