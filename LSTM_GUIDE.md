# LSTM Model Implementation Guide

## 1. Project Status Summary
**Status:** ✅ **Simulation Complete**, ✅ **Model Integrated**

| Component | Status | Description |
|-----------|--------|-------------|
| **Robotics Simulation** | **DONE** | Multi-drone navigation, circular coverage, and fault detection are fully operational. |
| **Drought Logic** | **DONE** | The *logic* for handling drought probabilities (risk allocation) is working. |
| **Prediction Model** | **DONE** | Trained **LSTM Neural Network** is now integrated and running live inference. |

---

## 2. Model Architecture
**Type:** Long Short-Term Memory (LSTM)
**Framework:** PyTorch
**Input Window:** 90 days (Lookback)
**Input Features:** 6 Meteorological variables (see below)
**Output:** Drought Probability (0.0 to 1.0)

### Input/Output Specifications
The model accepts a tensor of shape `(Batch_Size, 90, 6)`:

1.  **PRECTOT** (Precipitation)
2.  **QV2M** (Specific Humidity) -> **Proxy for Soil Moisture**
3.  **T2M_MAX** (Max Temperature)
4.  **T2M_MIN** (Min Temperature)
5.  **TS** (Earth Skin Temperature) -> **Proxy for Veg Stress**
6.  **PS** (Surface Pressure)

---

## 3. Integration & Usage
The model is already integrated into the simulation. No coding is required from the user.

### Files Location
*   **Weights:** `/home/ros/catkin_ws/src/multi_drone_sim/LSTM/lstm_model.pth`
*   **Class Def:** `/home/ros/catkin_ws/src/multi_drone_sim/LSTM/model.py`
*   **Inference:** `/home/ros/catkin_ws/src/multi_drone_sim/scripts/drought_probability_model.py`

### Dependencies
The simulation requires PyTorch to run the model. If PyTorch is missing, it falls back to a random simulation generator.

**Install Command (CPU-Optimized):**
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
```
*(This avoids installing the 700MB+ CUDA version which exceeds VM disk limits)*

### How to Run
Simply launch the exploration script:
```bash
roslaunch multi_drone_sim explore_areas.launch
```
Check the logs for: `[DroughtModel] Successfully loaded LSTM weights`.
