import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import DroughtDataset
from model import DroughtLSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model():
    # Parameters
    BATCH_SIZE = 4096
    SEQ_LENGTH = 90
    VAL_CSV = r'd:\Random Projects\BTP\dataset\validation_timeseries\validation_timeseries.csv'
    MODEL_PATH = 'lstm_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA NOT AVAILABLE. Running on CPU.")
        # Raise error if user specifically asked for GPU and we think we provided it?
        # For verification purposes, we want to know.
    
    # Load Dataset
    print("Loading Validation Dataset...")
    val_dataset = DroughtDataset(VAL_CSV, seq_length=SEQ_LENGTH, normalize=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load Model
    print("Loading Model...")
    model = DroughtLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=1).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    else:
        print(f"Warning: {MODEL_PATH} not found. Using untrained model for testing.")
        
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Move to CPU for metrics
            preds = outputs.cpu().numpy()
            targs = targets.numpy()
            
            all_predictions.extend(preds.flatten())
            all_targets.extend(targs)
            
    # Calculate Metrics
    # Remember targets were normalized (divided by 5.0)
    # We should scale them back to original 0-5 range for meaningful MAE
    
    y_true = np.array(all_targets) * 5.0
    y_pred = np.array(all_predictions) * 5.0
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print("Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
if __name__ == '__main__':
    import os
    evaluate_model()
