# LSTM Model Implementation Guide

## 1. Project Status Summary
**Status:** ✅ **Simulation Complete**, ⚠️ **Model Pending**

| Component | Status | Description |
|-----------|--------|-------------|
| **Robotics Simulation** | **DONE** | Multi-drone navigation, circular coverage, and fault detection are fully operational. |
| **Drought Logic** | **DONE** | The *logic* for handling drought probabilities (risk allocation) is working. |
| **Prediction Model** | **PENDING** | Currently using a "Heuristic Simulation". Needs to be replaced with a trained **LSTM Neural Network**. |

---

## 2. Research Context
We are implementing the architecture described in:
> **Paper:** *DroughtCast: A Machine Learning Forecast of the United States Drought Monitor*
> **Authors:** Colin Brust, Justin Kimball, et al. (Frontiers in Big Data, 2021)
> **DOI:** [10.3389/fdata.2021.773478](https://doi.org/10.3389/fdata.2021.773478)

**Dataset:** [US Drought Meteorological Data (Kaggle)](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data)

---

## 3. The Model You Need to Build

### 3.1 Architecture (LSTM)
You need to train a **Long Short-Term Memory (LSTM)** network. LSTMs are perfect for this because drought is a *temporal* phenomenon (today's drought depends on the last 90 days of rain).

*   **Type:** Sequence-to-One (Many-to-One)
*   **Input Window:** 90 days (Lookback)
*   **Target:** Drought Probability (0.0 to 1.0)

### 3.2 Imput/Output Specifications
To ensure the model works with our simulation, it **MUST** follow this signature:

**Input (Features):**
The model should accept a tensor of shape `(Batch_Size, 90, 6)`:
1.  **PRECTOT** (Precipitation)
2.  **GWETTOP** (or proxy e.g. **QV2M**) -> Soil Moisture/Humidity
3.  **T2M_MAX** (Max Temperature)
4.  **T2M_MIN** (Min Temperature)
5.  **NDVI** (Vegetation Index)
6.  **PS** (Surface Pressure)

6.  **PS** (Surface Pressure)

*Note: You MUST stick to 6 time-series inputs (Option A). Do NOT add static soil data (Option B) as it breaks the simulation integration.*
*Note: You can normalize these values (0-1) during preprocessing.*

**Output:**
A single float value `[0.0 - 1.0]` representing the **USDrought Monitor (USDM)** score normalized as a probability.
*   0.0 = No Drought (None)
*   1.0 = Exceptional Drought (D4)

---

## 4. How to Usage/Integration
Once you train this model on your host machine, save the weights as `lstm_model.pth`. Here is how we will plug it into the robot project:

### Step 1: Copy file
Place `lstm_model.pth` into:
`/home/ros/catkin_ws/src/multi_drone_sim/models/lstm_model.pth`

### Step 2: Update Code
We will modify `drought_probability_model.py` to use the real model instead of the heuristic.

```python
# In drought_probability_model.py

import torch
import torch.nn as nn

# 1. Define the architecture (MUST match your training code)
class DroughtLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(DroughtLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, 90, 6)
        out, _ = self.lstm(x)
        # Take last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# 2. Load the weights
def load_real_model():
    model = DroughtLSTM()
    model.load_state_dict(torch.load("models/lstm_model.pth"))
    model.eval()
    return model

# 3. Use in the pipeline
def predict_risk(csv_data_90_days):
    # Convert CSV data to Tensor
    input_tensor = preprocess(csv_data_90_days) 
    with torch.no_grad():
        probability = model(input_tensor).item()
    return probability
```

---

## 5. Summary of Instructions for You
1.  **Download Data:** Get the Kaggle dataset on your **Host Machine** (Laptop/Desktop).
2.  **Train:** Write a simple PyTorch/TensorFlow script to train an LSTM to predict 'score' from the 6 weather columns.
3.  **Save:** Identify the best model state and save it as `.pth`.
4.  **Transfer:** Send that `.pth` file to this VM.
5.  **Notify Me:** Tell me when the file is here, and I will write the integration code to swap out the simulation.
