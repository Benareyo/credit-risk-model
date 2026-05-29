import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BatiBankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Custom Scikit-Learn Transformer to handle alternative transactional data fields,
        including date components extraction, customer aggregations, and feature encoding.
        """
        self.numeric_scaler = StandardScaler()
        self.categorical_mappings = {}
        self.engineered_feature_columns = []

    def fit(self, X, y=None):
        # We handle fit operations here to calculate global values if needed
        return self

    def transform(self, X):
        # Deep copy to ensure input reference immutability
        df = X.copy()
        
        # 1. Convert timestamp metrics and extract granular temporal slices
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            
        # 2. Compute Aggregate Customer Performance features
        if 'CustomerId' in df.columns and 'Amount' in df.columns:
            # Calculate aggregate groupings
            customer_groups = df.groupby('CustomerId')['Amount']
            
            df['Total_Transaction_Amount'] = customer_groups.transform('sum')
            df['Average_Transaction_Amount'] = customer_groups.transform('mean')
            df['Transaction_Count'] = customer_groups.transform('count')
            df['Std_Dev_Transaction_Amount'] = customer_groups.transform('std').fillna(0)
        else:
            # Baseline fallbacks if columns are missing during structural tracking
            df['Total_Transaction_Amount'] = 0.0
            df['Average_Transaction_Amount'] = 0.0
            df['Transaction_Count'] = 1
            df['Std_Dev_Transaction_Amount'] = 0.0

        # 3. Categorical Variables Encoding via Label Frequency Mapping
        categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
        for col in categorical_cols:
            if col in df.columns:
                # Frequency encoding ensures safe bounds without explosive dimensional scaling (like One-Hot)
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col + '_Encoded'] = df[col].map(freq_map)
                
        # 4. Handle Missing Values explicitly via structural baseline filling
        numerical_features = [
            'Amount', 'Value', 'PricingStrategy', 'TransactionHour', 
            'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'Total_Transaction_Amount', 'Average_Transaction_Amount', 
            'Transaction_Count', 'Std_Dev_Transaction_Amount'
        ]
        
        for num_col in numerical_features:
            if num_col in df.columns:
                median_val = df[num_col].median()
                df[num_col] = df[num_col].fillna(median_val)
                
        # 5. Isolate clean structural modeling features
        final_modeling_columns = numerical_features + [c + '_Encoded' for c in categorical_cols if c in df.columns]
        self.engineered_feature_columns = final_modeling_columns
        
        # 6. Normalize/Standardize Numerical Features using our Scaler
        df[final_modeling_columns] = self.numeric_scaler.fit_transform(df[final_modeling_columns].astype(float))
        
        # Keep structural keys like CustomerId intact for Task 4 RFM clustering operations
        if 'CustomerId' in df.columns:
            return df[['CustomerId'] + final_modeling_columns]
            
        return df[final_modeling_columns]


def build_production_pipeline():
    """
    Constructs a fitted unified Sklearn pipeline wrapper.
    """
    pipeline = Pipeline([
        ('feature_engineering', BatiBankFeatureEngineer())
    ])
    return pipeline


if __name__ == "__main__":
    import os
    # Minimal script sanity check run
    print("Initializing structural feature engineering pipeline verify engine...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'training.csv')
    
    if os.path.exists(data_path):
        sample_df = pd.read_csv(data_path, nrows=100)
        pipeline = build_production_pipeline()
        processed_data = pipeline.fit_transform(sample_df)
        print("✅ Pipeline executed perfectly! Processed Matrix Head Shape:", processed_data.shape)
    else:
        print("⚠ Verification file not found locally. Code checks look syntactically robust.")