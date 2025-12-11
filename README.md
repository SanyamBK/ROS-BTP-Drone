# LSTM Drought Prediction Model

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict drought severity based on meteorological time-series data. The model is designed to integrate with a larger Robotics/Drone simulation framework.

**Dataset**: [US Drought Meteorological Data (Kaggle)](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data?select=test_timeseries)

## Model Architecture
- **Type**: Sequence-to-One (Many-to-One) LSTM
- **Framework**: PyTorch
- **Input Shape**: `(Batch_Size, 90, 6)`
- **Output Shape**: `(Batch_Size, 1)` (Drought Probability 0.0 - 1.0)

### Layers
1.  **LSTM Layer**: 
    - Input Size: 6
    - Hidden Size: 64
    - Num Layers: 2
    - Batch First: True
2.  **Fully Connected Layer**: `Linear(64 -> 1)`
3.  **Activation**: `Sigmoid` (Output scaled 0-1)

## Data Specifications
### Input Features (6 Channels)
The model expects a sequence of **90 days** of historical data with the following 6 meteorological features:

1.  **PRECTOT**: Precipitation
2.  **QV2M**: Specific Humidity at 2 Meters (Proxy for Soil Moisture/GWETTOP)
3.  **T2M_MAX**: Maximum Temperature at 2 Meters
4.  **T2M_MIN**: Minimum Temperature at 2 Meters
5.  **TS**: Skin Temperature (Proxy for Vegetation/NDVI)
6.  **PS**: Surface Pressure

*Note: Features `GWETTOP` and `NDVI` from the original guide were substituted with `QV2M` and `TS` respectively due to dataset availability, ensuring the input shape remains consistent.*

### Target Variable
- **Score**: US Drought Monitor (USDM) score (0 to 5 integers).
- **Normalization**: The target is divided by `5.0` to normalize it to the range `[0, 1]`.

## Preprocessing Pipeline
1.  **Interpolation**: Missing values (NaNs) in the dataset are filled using **Linear Interpolation**.
2.  **Normalization (Features)**: Standard Scaler (Mean subtraction, division by Std Dev).
3.  **Sliding Window**: Sequences of 90 days are generated with a stride of 1 day. A sequence is only valid if it falls entirely within a single FIPS (county) code.

## Training Details
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (Learning Rate: 0.001)
- **Batch Size**: 4096
- **Epochs**: 10
- **Best Validation Loss**: ~0.025 (approx. 0.68 MSE on original scale)

## Usage
### Inference (Loading the Model)
```python
import torch
import torch.nn as nn

class DroughtLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(DroughtLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load Weights
model = DroughtLSTM()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Predict (Variable 'input_tensor' must be shape (1, 90, 6))
with torch.no_grad():
    risk_score = model(input_tensor).item() # Returns float 0.0 - 1.0
```

### Files
- `train.py`: Main training loop.
- `evaluate.py`: Evaluation script (calculates MSE/MAE).
- `dataset_loader.py`: Handles data loading, windowing, and interpolation.
- `model.py`: Defines the PyTorch class.
- `lstm_model.pth`: Saved model weights.
