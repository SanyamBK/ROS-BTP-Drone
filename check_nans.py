import pandas as pd
import numpy as np

CSV_PATH = r'd:\Random Projects\BTP\dataset\train_timeseries\train_timeseries.csv'

def check_data():
    print(f"Checking {CSV_PATH} for NaNs...")
    # Load first 100k rows to be fast
    df = pd.read_csv(CSV_PATH, nrows=500000)
    
    print("Columns:", df.columns)
    print("Head:\n", df.head())
    
    nan_counts = df.isna().sum()
    print("\nNaN Counts (first 500k rows):")
    print(nan_counts)
    
    # Check specifically score
    score_nans = df['score'].isna().sum()
    print(f"\nScore NaNs: {score_nans}/{len(df)}")
    
    if score_nans > 0:
        print("CRITICAL: Target 'score' has missing values.")
        
    # Check features
    feats = ['PRECTOT', 'PS', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'TS']
    feat_nans = df[feats].isna().sum().sum()
    print(f"Feature NaNs: {feat_nans}")

if __name__ == '__main__':
    check_data()
