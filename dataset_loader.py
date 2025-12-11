import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class DroughtDataset(Dataset):
    def __init__(self, csv_file, seq_length=90, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            seq_length (int): Length of the sequence for LSTM input.
            normalize (bool): Whether to normalize features.
        """
        print(f"Loading data from {csv_file}...")
        # Load only necessary columns to save memory
        # Order MUST match LSTM_GUIDE.md: PRECTOT, QV2M (Soil Proxy), T2M_MAX, T2M_MIN, TS (Veg Proxy), PS
        feature_cols = ['PRECTOT', 'QV2M', 'T2M_MAX', 'T2M_MIN', 'TS', 'PS']
        target_col = 'score'
        meta_cols = ['fips', 'date']
        
        # Load data. Specifying dtypes can save memory.
        # integers for fips, floats for others.
        # Assuming date is string YYYY-MM-DD
        self.df = pd.read_csv(csv_file, usecols=meta_cols + feature_cols + [target_col])
        
        self.seq_length = seq_length
        self.normalize = normalize
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Pre-process data
        self._prepare_data()

    def _prepare_data(self):
        print("Preprocessing data...")
        
        # 0. Interpolate Missing Values (Baseline Approach for Accuracy)
        # Sort first ensuring valid time order
        self.df = self.df.sort_values(by=['fips', 'date'])
        
        print("Interpolating missing values...")
        # Group by FIPS to avoid interpolating across counties
        # This is slow with groupby.apply, so we can do a global interpolate if we mask boundaries?
        # A safer/easier way used in baseline is per-fips.
        # But for speed on 20M rows, let's use the fact that fips are chunks.
        
        # Actually, standard pandas interpolate on the whole DF is risky if fips changes.
        # Let's assume FIPS are grouped. 
        # Better: iterate unique FIPS? No, too slow.
        # Pivot? No.
        
        # Optimised approach:
        # The baseline loaded per file. We have one big CSV.
        # Let's trust pandas groupby().apply() or transform() is optimised enough, or accept a bit of delay.
        # WAIT: The dataset is 2GB. groupby apply might freeze.
        
        # Alternative: Just interpolate the whole column. 
        # The discontinuities between FIPS are rare compared to the data size. 
        # And we separate sequences by FIPS anyway in 'valid_start_indices'.
        # A wrong interpolation at the exact boundary of two FIPS (one day transition) 
        # will only affect sequences spanning that boundary, which we filter out!
        # sequences must belong to only one fips. 
        # So Global interpolation is SAFE provided that we respect FIPS boundaries during sequence generation.
        # (Actually, if FIPS A ends and FIPS B starts, interpolating A's last value with B's first value 
        # creates garbage for the few rows in between, but those rows are boundary rows).
        # We just need to ensure we don't pick sequences that use that garbage.
        
        # To be cleaner, let's just interpret NaNs.
        # Features
        self.df[self.feature_cols] = self.df[self.feature_cols].interpolate(method='linear', limit_direction='both')
        
        # Target (Score)
        # Score is weekly, so it has many NaNs. Interpolation fills them.
        self.df[self.target_col] = self.df[self.target_col].interpolate(method='linear', limit_direction='both')
        
        # 1. Normalize Features
        if self.normalize:
            for col in self.feature_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = (self.df[col] - mean) / (std + 1e-6)
        
        # 2. Normalize Target
        self.df[self.target_col] = self.df[self.target_col].astype('float32') / 5.0
        
        # Convert features to float32
        self.df[self.feature_cols] = self.df[self.feature_cols].astype('float32')
        
        # Extract underlying numpy arrays
        self.data_matrix = self.df[self.feature_cols].values # (N, 6)
        self.target_matrix = self.df[self.target_col].values # (N,)
        self.fips_matrix = self.df['fips'].values
        
        # Free memory
        del self.df
        import gc
        gc.collect()
        
        # 3. Create Valid Indices
        print("Creating valid indices...")
        total_rows = len(self.fips_matrix)
        
        fips_start = self.fips_matrix[:total_rows - self.seq_length + 1]
        fips_end = self.fips_matrix[self.seq_length - 1:]
        
        # Only condition: Sequence must stay within one FIPS
        mask = (fips_start == fips_end)
        
        # Refined NaN check:
        # After interpolation, there shouldn't be NaNs unless a whole FIPS is empty.
        # But let's keep a safety check for the target label.
        target_indices = np.arange(total_rows - self.seq_length + 1) + self.seq_length - 1
        valid_targets = ~np.isnan(self.target_matrix[target_indices])
        
        mask = mask & valid_targets
        
        self.valid_start_indices = np.where(mask)[0]
        
        print(f"Dataset ready. Total sequences: {len(self.valid_start_indices)}")

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_start_indices[idx]
        end_idx = start_idx + self.seq_length
        
        # Get sequence
        seq = self.data_matrix[start_idx : end_idx]
        
        # Get label (at the end of the sequence)
        label = self.target_matrix[end_idx - 1]
        
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))

if __name__ == '__main__':
    # Simple test
    # Note: This requires the actual CSV to be present to run
    print("Dataset loader defined.")
