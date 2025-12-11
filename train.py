import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DroughtDataset
from model import DroughtLSTM
import os
import time

def train_model():
    # Parameters
    BATCH_SIZE = 4096 # Increased for speed on GPU
    EPOCHS = 10 
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 90
    NUM_WORKERS = 0 # Set to 0 to avoid Windows multiprocessing/pickling error with large dataset
    
    # Paths (adjust as needed)
    TRAIN_CSV = r'd:\Random Projects\BTP\dataset\train_timeseries\train_timeseries.csv'
    VAL_CSV = r'd:\Random Projects\BTP\dataset\validation_timeseries\validation_timeseries.csv'
    MODEL_SAVE_PATH = 'lstm_model.pth'

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Datasets
    print("Initializing Training Dataset...")
    train_dataset = DroughtDataset(TRAIN_CSV, seq_length=SEQ_LENGTH, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Load Validation Dataset
    # NOTE: In a real scenario, we should use the same normalization statistics as training.
    # The current Dataset class re-calculates stats on the input file. 
    # For now, we accept this simplification, or we could modify Dataset to accept stats.
    print("Initializing Validation Dataset...")
    val_dataset = DroughtDataset(VAL_CSV, seq_length=SEQ_LENGTH, normalize=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Initialize Model
    model = DroughtLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # Training
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # targets shape: (batch), unsqueeze to (batch, 1)
            targets = targets.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")

    print("Training Complete.")

if __name__ == '__main__':
    train_model()
