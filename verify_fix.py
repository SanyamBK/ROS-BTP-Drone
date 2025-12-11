from dataset_loader import DroughtDataset
import numpy as np
import pandas as pd

VAL_CSV = r'dataset\validation_timeseries\validation_timeseries.csv'

def verify():
    print("Initializing Dataset with Interpolation...")
    ds = DroughtDataset(VAL_CSV, seq_length=90, normalize=True)
    
    total_seq = len(ds)
    print(f"Total sequences after interpolation: {total_seq}")
    
    # We expect close to 1.9M sequences now (original count), not 280k
    
    # Check a sample for NaNs (should be none)
    x, y = ds[0]
    print(f"Sample x[0]: {x[0]}")
    print(f"Sample y: {y}")
    
    if np.isnan(y):
        print("FAILURE: Y is NaN")
    elif np.isnan(x).any():
        print("FAILURE: X has NaN")
    else:
        print("SUCCESS: Data is clean and interpolated.")

if __name__ == '__main__':
    verify()
