import pandas as pd
import numpy as np
import os
import torch
from dataset_loader import DroughtDataset
from model import DroughtLSTM
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def run_test():
    print("Starting verification test...")
    
    # 1. Create Dummy Data
    csv_path = 'test_dummy_data.csv'
    num_rows = 200 # Enough for 2 sequences of 90 with stride
    dates = pd.date_range(start='2020-01-01', periods=num_rows)
    
    # Cols: fips, date, PRECTOT, PS, T2M_MAX, T2M_MIN, QV2M, TS, score
    data = {
        'fips': [1001] * num_rows,
        'date': dates,
        'PRECTOT': np.random.rand(num_rows),
        'PS': np.random.rand(num_rows),
        'T2M_MAX': np.random.rand(num_rows),
        'T2M_MIN': np.random.rand(num_rows),
        'QV2M': np.random.rand(num_rows),
        'TS': np.random.rand(num_rows),
        'score': np.random.randint(0, 6, num_rows) # 0-5
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Created dummy dataset at {csv_path}")
    
    try:
        # 2. Test Dataset Loader
        print("Testing Dataset Loader...")
        dataset = DroughtDataset(csv_path, seq_length=90, normalize=True)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Dataset empty! Logic check needed.")
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        sample_x, sample_y = next(iter(loader))
        print("Sample input shape:", sample_x.shape) # (4, 90, 6)
        print("Sample target shape:", sample_y.shape) # (4,)
        
        # 3. Test Model
        print("Testing Model...")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is NOT available. Test will run on CPU, but GPU training was requested.")
        else:
            print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DroughtLSTM().to(device)
        # sample_x is already a tensor from DataLoader
        sample_x = sample_x.to(dtype=torch.float32).to(device)
        output = model(sample_x)
        print("Model output shape:", output.shape) # (4, 1)
        
        # 4. Test Training Step
        print("Testing Training Step...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        target = sample_y.unsqueeze(1).float().to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Training step successful. Loss: {loss.item()}")
        
        print("VERIFICATION SUCCESSFUL: Pipeline is functional.")
        
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print("Cleaned up dummy file.")

if __name__ == '__main__':
    run_test()
