import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class BatiBankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='FraudResult'):
        """
        Comprehensive Scikit-Learn Transformer implementing:
        1. Temporal feature extraction
        2. Customer aggregation metrics
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
        
        if self.target_col not in df.columns:
            np.random.seed(42)
            df[self.target_col] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
            
        y_target = df[self.target_col]
        total_goods = (y_target == 0).sum()
        total_bads = (y_target == 1).sum()
        
        if total_goods == 0: total_goods = 1
        if total_bads == 0: total_bads = 1

        cols_to_woe = ['ProductCategory', 'ChannelId']
        
        for col in cols_to_woe:
            if col in df.columns:
                self.woe_maps[col] = {}
                grouped = df.groupby(col)[self.target_col].agg(['count', 'sum'])
                grouped.columns = ['Total', 'Bads']
                grouped['Goods'] = grouped['Total'] - grouped['Bads']
                
                for cat, row in grouped.iterrows():
                    g_dist = row['Goods'] / total_goods
                    b_dist = row['Bads'] / total_bads
                    
                    if g_dist == 0: g_dist = 0.0001
                    if b_dist == 0: b_dist = 0.0001
                    
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
                df[col + '_WoE'] = df[col].map(mapping).fillna(0.0)

        # 5. Handle Missing Values explicitly via structural baseline filling
        numerical_features = [
            'Amount', 'Value', 'PricingStrategy', 'TransactionHour', 
            'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'Total_Transaction_Amount', 'Average_Transaction_Amount', 
            'Transaction_Count', 'Std_Dev_Transaction_Amount'
        ]
        
        for num_col in numerical_features:
            if num_col in df.columns:
                df[num_col] = df[num_col].fillna(df[num_col].median() if len(df) > 0 else 0.0)

        # 6. Normalize/Standardize numerical inputs to scale uniformly
        if len(df) > 0:
            df[numerical_features] = self.numeric_scaler.fit_transform(df[numerical_features].astype(float))

        final_cols = numerical_features + [c + '_Encoded' for c in categorical_cols if c in df.columns]
        final_cols += [c + '_WoE' for c in self.woe_maps.keys() if c in df.columns]
        self.final_features = final_cols

        if 'CustomerId' in df.columns:
            return df[['CustomerId'] + final_cols]
        return df[final_cols]


def build_production_pipeline():
    """Constructs a clean, standalone Pipeline object."""
    pipeline = Pipeline([
        ('feature_engineering', BatiBankFeatureEngineer())
    ])
    return pipeline


def engineer_proxy_target_variable(raw_df, random_state=42):
    """Calculates robust RFM profiles and tags high-risk clusters via KMeans."""
    print("Calculating robust RFM metrics per customer...")
    df = raw_df.copy()
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    df['Abs_Amount'] = df['Amount'].abs()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Abs_Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Abs_Amount': 'Monetary'
    })
    
    rfm_log = np.log1p(rfm)
    
    for col in rfm_log.columns:
        if rfm_log[col].isnull().any() or np.isinf(rfm_log[col]).any():
            median_val = rfm_log[col].replace([np.inf, -np.inf], np.nan).median()
            rfm_log[col] = rfm_log[col].fillna(median_val if pd.notnull(median_val) else 0.0)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    cluster_monetary_means = rfm.groupby('Cluster')['Monetary'].mean()
    high_risk_cluster = cluster_monetary_means.idxmin()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    print(f"Proxy variable assignment complete! High-risk cluster identified as Cluster {high_risk_cluster}.")
    print(rfm['is_high_risk'].value_counts())
    
    return rfm[['is_high_risk']].to_dict()['is_high_risk']


def generate_and_save_processed_dataset(raw_data_path, output_dir):
    """Executes transformations and saves the finalized dataset."""
    raw_df = pd.read_csv(raw_data_path)
    risk_mapping = engineer_proxy_target_variable(raw_df)
    
    pipeline = build_production_pipeline()
    processed_features = pipeline.fit_transform(raw_df)
    
    processed_features['is_high_risk'] = processed_features['CustomerId'].map(risk_mapping)
    processed_features['is_high_risk'] = processed_features['is_high_risk'].fillna(0).astype(int)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'processed_credit_data.csv')
    processed_features.to_csv(save_path, index=False)
    print(f"✅ Master Processed dataset saved perfectly to: {save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', 'raw', 'training.csv')
    output_directory = os.path.join(base_dir, 'data', 'processed')
    
    if os.path.exists(input_file):
        generate_and_save_processed_dataset(input_file, output_directory)
    else:
        print(f"⚠ Missing source training file path location at: {input_file}")