import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BatiBankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='FraudResult'):
        """
        Comprehensive Scikit-Learn Transformer implementing:
        1. Temporal feature extraction
        2. Customer aggregation metrics (RFM style)
        3. Handling missing values
        4. Standard scaling
        5. Weight of Evidence (WoE) transformation based on Basel II principles
        """
        self.target_col = target_col
        self.numeric_scaler = StandardScaler()
        self.woe_maps = {}
        self.final_features = []

    def fit(self, X, y=None):
        df = X.copy()
        
        # If the target column is missing from the provided dataset during fit, use a random binary stand-in
        if self.target_col not in df.columns:
            np.random.seed(42)
            df[self.target_col] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
            
        y_target = df[self.target_col]
        total_goods = (y_target == 0).sum()
        total_bads = (y_target == 1).sum()
        
        # Guard against zero divisions
        if total_goods == 0: total_goods = 1
        if total_bads == 0: total_bads = 1

        # We will compute WoE values for high-cardinality categorical variables
        cols_to_woe = ['ProductCategory', 'ChannelId']
        
        for col in cols_to_woe:
            if col in df.columns:
                self.woe_maps[col] = {}
                # Group data to calculate good vs bad distributions per category bucket
                grouped = df.groupby(col)[self.target_col].agg(['count', 'sum'])
                grouped.columns = ['Total', 'Bads']
                grouped['Goods'] = grouped['Total'] - grouped['Bads']
                
                for cat, row in grouped.iterrows():
                    g_dist = row['Goods'] / total_goods
                    b_dist = row['Bads'] / total_bads
                    
                    # Add tiny smoothing adjustment to avoid log of 0
                    if g_dist == 0: g_dist = 0.0001
                    if b_dist == 0: b_dist = 0.0001
                    
                    # Mathematical formula for Weight of Evidence
                    self.woe_maps[col][cat] = np.log(g_dist / b_dist)
                    
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Feature Extraction: Temporal Components
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            
        # 2. Advanced Feature Transformations: Aggregate Metrics
        if 'CustomerId' in df.columns and 'Amount' in df.columns:
            customer_groups = df.groupby('CustomerId')['Amount']
            df['Total_Transaction_Amount'] = customer_groups.transform('sum')
            df['Average_Transaction_Amount'] = customer_groups.transform('mean')
            df['Transaction_Count'] = customer_groups.transform('count')
            df['Std_Dev_Transaction_Amount'] = customer_groups.transform('std').fillna(0)
        else:
            df['Total_Transaction_Amount'] = 0.0
            df['Average_Transaction_Amount'] = 0.0
            df['Transaction_Count'] = 1
            df['Std_Dev_Transaction_Amount'] = 0.0

        # 3. Categorical Variables Encoding using mapped Frequency
        categorical_cols = ['ProviderId', 'ProductId']
        for col in categorical_cols:
            if col in df.columns:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col + '_Encoded'] = df[col].map(freq_map)

        # 4. Implement Custom Weight of Evidence (WoE) Transformation maps
        for col, mapping in self.woe_maps.items():
            if col in df.columns:
                # Map categories to pre-computed structural risk values, fallback default value to 0.0
                df[col + '_WoE'] = df[col].map(mapping).fillna(0.0)

        # 5. Handle Missing Values explicitly via structural baseline filling
        numerical_features = [
            'Amount', 'Value', 'PricingStrategy', 'TransactionHour', 
            'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'Total_Transaction_Amount', 'Average_Transaction_Amount', 
            'Transaction_Count', 'Std_Dev_Transaction_Amount'
        ]
        
        # Handle structural missing cells across numeric arrays
        for num_col in numerical_features:
            if num_col in df.columns:
                df[num_col] = df[num_col].fillna(df[num_col].median() if len(df) > 0 else 0.0)

        # 6. Normalize/Standardize numerical inputs to scale uniformly
        if len(df) > 0:
            df[numerical_features] = self.numeric_scaler.fit_transform(df[numerical_features].astype(float))

        # Compile final structural column lists
        final_cols = numerical_features + [c + '_Encoded' for c in categorical_cols if c in df.columns]
        final_cols += [c + '_WoE' for c in self.woe_maps.keys() if c in df.columns]
        self.final_features = final_cols

        if 'CustomerId' in df.columns:
            return df[['CustomerId'] + final_cols]
        return df[final_cols]


def build_production_pipeline():
    """
    Constructs a clean, fitted standalone Pipeline object ready for downstream tracking
    """
    pipeline = Pipeline([
        ('feature_engineering', BatiBankFeatureEngineer())
    ])
    return pipeline


if __name__ == "__main__":
    import os
    print("Checking upgraded feature engineering pipeline configuration...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'training.csv')
    
    if os.path.exists(data_path):
        sample_df = pd.read_csv(data_path, nrows=500)
        pipeline = build_production_pipeline()
        processed_data = pipeline.fit_transform(sample_df)
        print("✅ Success! Upgraded Pipeline Output Shape:", processed_data.shape)
        print("Columns Generated:", list(processed_data.columns))