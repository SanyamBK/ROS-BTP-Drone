# Drought Monitoring System - Implementation Guide

## 📋 Overview

This is a complete implementation of an **intelligent multi-drone drought monitoring system** based on the research paper:

**"DroughtCast: A Machine Learning Forecast of the United States Drought Monitor"**  
Authors: Colin Brust, et al. (Frontiers in Big Data, 2021)

The system demonstrates:
- ✅ **Drought probability estimation** (random probabilities for rapid testing)
- ✅ **Area prioritization** based on risk levels
- ✅ **Dynamic drone allocation** with coverage guarantees
- ✅ **Sensor fault detection** using statistical hypothesis testing
- ✅ **Multi-drone verification** with probability merging
- ✅ **Auditor deployment** for validating faulty sensors

---

## 🎯 Key Modules

### 1. **Drought Probability Model** (`drought_probability_model.py`)

Generates drought risk estimates for farmland areas.

#### Key Features:
- Pre-generated probability pool (5%-95% range) for testing without ML training
- Feature-based probability calculation using weighted indices:
  - Rainfall Deficit (SPI) - 25% weight
  - Soil Moisture Deficit (SMI) - 25% weight
  - Vegetation Stress (VCI) - 20% weight
  - Heatwave Intensity (TCI) - 15% weight
  - Drought Frequency - 10% weight
  - Trend Coefficient - 5% weight

#### Usage:
```python
from drought_probability_model import DroughtProbabilityModel

model = DroughtProbabilityModel(seed=42)

# Get random probabilities for 10 areas
probs = model.get_area_probabilities(10)
# Returns: {'area_1': 0.65, 'area_2': 0.35, ...}

# Calculate from features (ML would train this, we use weighted formula)
features = {
    'rain_deficit': 0.8,
    'soil_deficit': 0.7,
    'veg_stress': 0.6,
    'heat_index': 0.5,
    'drought_freq': 0.4,
    'trend': 0.3
}
prob = model.calculate_probability_from_features(features)
# Returns: 0.72 (logistic mapping to 0.05-0.95 range)

# Extract features from Kaggle dataset
features = model.extract_features_from_csv(
    '/path/to/meteorological/data.csv',
    lookback_days=90
)
```

#### Logistic Mapping Formula:
```
P(drought) = 0.05 + 0.90 / (1 + exp(-10 * (score - 0.5)))
```

---

### 2. **Sensor Fault Detection** (`sensor_fault_detection.py`)

Detects faulty drone sensors and merges corrected measurements.

#### Key Components:

**SensorFaultDetector:**
- Compares model predictions with drone sensor readings
- Statistical hypothesis testing: fault if |error| > 2σ + threshold
- Calculates confidence in fault detection

**SensorFusion:**
- Inverse-variance weighting for optimal multi-sensor fusion
- Bayesian inference option for probability updates
- Reliability-weighted fusion using drone track records

**DroneVerificationSystem:**
- Manages all sensor measurements per area
- Triggers auditor deployment on fault detection
- Merges auditor verifications with original readings

#### Usage:
```python
from sensor_fault_detection import (
    SensorFaultDetector, SensorFusion, DroneVerificationSystem,
    SensorReading
)

# Detect faulty sensor
detector = SensorFaultDetector(fault_threshold=0.15, sensitivity=2.0)
model_prob = 0.65
sensor_prob = 0.25  # Clearly wrong
sensor_noise = 0.05

is_faulty, reason, confidence = detector.detect_fault(
    model_prob, sensor_prob, sensor_noise
)
# Returns: (True, "Deviation exceeds threshold", 0.89)

# Fuse multiple sensor readings
readings = [
    SensorReading(drone_id=1, area_id="area_1", probability=0.55, 
                 noise_std=0.05, timestamp=0.0),
    SensorReading(drone_id=2, area_id="area_1", probability=0.62, 
                 noise_std=0.03, timestamp=1.0),
]
fused_prob, fused_std = SensorFusion.inverse_variance_fusion(readings)
# Returns: (0.60, 0.025) - weighted average favoring more precise sensor

# Manage verification and auditors
system = DroneVerificationSystem()
system.add_measurement(reading, model_probability=0.65)
# Automatically detects faults and logs them

# Merge auditor verification
fused = system.merge_auditor_verification(
    area_id="area_1",
    original_drone_id=1,
    auditor_id=10,
    auditor_reading=0.67,
    auditor_noise=0.03
)
# Returns: 0.65 (corrected probability)
```

#### Inverse-Variance Fusion Formula:
```
Weight_i = 1 / σ_i²
P_fused = Σ(Weight_i × P_i) / Σ(Weight_i)
σ_fused = 1 / √(Σ(Weight_i))
```

---

### 3. **Area Allocation** (`area_allocation.py`)

Prioritizes areas and allocates drones dynamically.

#### Key Components:

**AreaPrioritizer:**
- Ranks areas by drought probability
- Categorizes into HIGH (>70%), MEDIUM (40-70%), LOW (<40%)
- Calculates coverage needs per area
- Tracks historical probability trends

**DynamicDroneAllocator:**
- Implements 5-phase allocation algorithm
- Guarantees minimum 1 drone per area
- Prioritizes high-risk areas for additional coverage
- Maintains emergency reserves

#### Allocation Algorithm:
1. **Phase 1:** Assign 1 explorer to each area (sorted by risk)
2. **Phase 2:** Deploy auditors to fault-suspected areas
3. **Phase 3:** Add drones to HIGH-risk areas (up to max)
4. **Phase 4:** Add drones to MEDIUM-risk areas (if space)
5. **Phase 5:** Keep remaining as emergency reserves

#### Usage:
```python
from area_allocation import (
    AreaPrioritizer, DynamicDroneAllocator, Area, Drone
)

# Create areas
areas = [
    Area(area_id="area_1", drought_probability=0.85, size_m2=12000,
         coverage_radius=60, x_center=-12, y_center=9, crop_type="Wheat"),
    Area(area_id="area_2", drought_probability=0.45, size_m2=8000,
         coverage_radius=50, x_center=-2, y_center=11.5, crop_type="Corn"),
]

# Create drones
drones = [Drone(drone_id=i, sensor_noise=0.05) for i in range(10)]

# Prioritize areas
prioritizer = AreaPrioritizer()
ranked = prioritizer.rank_areas_by_risk(areas)
# Returns: [area_1 (0.85), area_2 (0.45)] sorted by risk

# Categorize by risk level
categories = prioritizer.categorize_areas_by_risk_level(areas)
# Returns: {'high': [area_1], 'medium': [], 'low': [area_2]}

# Allocate drones
allocator = DynamicDroneAllocator(
    total_drones=10,
    total_areas=2,
    min_drones_per_area=1,
    max_drones_per_area=3,
    reserve_percentage=0.1
)

result = allocator.allocate_drones(areas, drones)
# Returns:
# - allocations: {"area_1": [0,1,2], "area_2": [3]}
# - reserve_drones: [4,5]
# - allocation_summary with statistics
```

---

### 4. **Integrated Demo** (`integrated_demo.py`)

Complete end-to-end demonstration of the system.

#### Workflow:
1. **Initialize:** Create 10 areas and 18 drones
2. **Allocate:** Assign drones to areas by risk
3. **Simulate:** Generate sensor measurements (some faulty)
4. **Detect:** Identify faulty sensors
5. **Verify:** Deploy auditors to validate
6. **Report:** Generate comprehensive mission summary

#### Running the Demo:
```bash
python3 scripts/integrated_demo.py
```

#### Expected Output:
```
PHASE 1: AREA PRIORITIZATION & DRONE ALLOCATION
  Area Risk Ranking:
    Rank 1: area_1 (Wheat, 0.850) - HIGH
    Rank 2: area_4 (Wheat, 0.720) - HIGH
    ...
  Allocation Summary:
    Total drones allocated: 17
    Drones in reserve: 1
    Areas covered: 10/10

PHASE 2: MISSION EXECUTION & SENSOR MEASUREMENTS
  area_1 (Risk: 0.850, Crop: Wheat):
    Drone 0: 0.841 (GOOD, σ=0.0523)
    Drone 1: 0.623 (FAULTY, σ=0.0416)
    ...

PHASE 3: AUDITOR VERIFICATION & SENSOR FUSION
  area_1: Deploying auditor to verify drone 1
    Auditor 101 reading: 0.848 (σ=0.0200)
    [SENSOR FUSION] Area area_1 verification complete
      Original (faulty) reading: 0.623 (σ=0.0416)
      Auditor reading: 0.848 (σ=0.0200)
      Fused probability: 0.821
      Confidence: 0.957

MISSION REPORT & SUMMARY
  System Configuration:
    Total drones: 18
    Total areas: 10
    Average drones per area: 1.8
  ...
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PROBABILITY GENERATION & PRIORITIZATION           │
└─────────────────────────────────────────────────────────────┘
  Kaggle Dataset → DroughtProbabilityModel → Area Probabilities
                                                      ↓
                                    AreaPrioritizer → Risk Ranking
                                                      ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: DRONE ALLOCATION                                  │
└─────────────────────────────────────────────────────────────┘
  Drones × Ranked Areas → DynamicDroneAllocator → Allocations
                                                      ↓
                      area_1: [drone_0, drone_1, drone_2]
                      area_2: [drone_3, drone_4]
                      area_3: [drone_5]
                      reserve: [drone_6, drone_7, ...]
                                                      ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: MISSION EXECUTION & MEASUREMENT                    │
└─────────────────────────────────────────────────────────────┘
  Drones Measure → SensorReading Objects → DroneVerificationSystem
                                                      ↓
                       Model Prob: 0.65
                       Sensor 1:   0.63 ✓
                       Sensor 2:   0.25 ✗ (FAULTY)
                                                      ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: FAULT DETECTION & AUDITOR DEPLOYMENT               │
└─────────────────────────────────────────────────────────────┘
  SensorFaultDetector → Is Faulty? → Deploy Auditor from Reserve
                                                      ↓
                       Auditor Measures: 0.66
                                                      ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: SENSOR FUSION & CORRECTED PROBABILITY              │
└─────────────────────────────────────────────────────────────┘
  SensorFusion.inverse_variance_fusion(
    [Faulty(0.25, σ=0.05), Auditor(0.66, σ=0.02)]
  ) → Fused: 0.64 (weighted toward low-noise auditor)
```

---

## 🔧 Configuration Parameters

### Fault Detection (`sensor_fault_detection.py`):
```python
SensorFaultDetector(
    fault_threshold=0.15,      # Max deviation (15%)
    sensitivity=2.0            # Sensitivity multiplier (1.5-2.5)
)
```

### Drone Allocation (`area_allocation.py`):
```python
DynamicDroneAllocator(
    total_drones=18,
    total_areas=10,
    min_drones_per_area=1,     # Minimum coverage
    max_drones_per_area=3,     # Maximum per area
    reserve_percentage=0.1     # 10% kept as emergency
)
```

### Drought Model (`drought_probability_model.py`):
```python
DroughtProbabilityModel(
    seed=42,                   # Reproducibility
    weights={
        'rain_deficit': 0.25,
        'soil_deficit': 0.25,
        'veg_stress': 0.20,
        'heat_index': 0.15,
        'drought_freq': 0.10,
        'trend': 0.05
    }
)
```

---

## 📈 System Metrics

After allocation and verification, the system reports:

1. **Coverage Metrics:**
   - Total drones allocated
   - Drones in reserve
   - Areas covered percentage
   - Average drones per area

2. **Risk Distribution:**
   - HIGH-risk areas (>70% probability)
   - MEDIUM-risk areas (40-70%)
   - LOW-risk areas (<40%)

3. **Verification Results:**
   - Number of faults detected
   - Number of auditor deployments
   - Average verification confidence
   - Fused probability estimates

---

## 🧪 Testing

Run individual module tests:

```bash
# Test drought probability model
python3 scripts/drought_probability_model.py

# Test fault detection
python3 scripts/sensor_fault_detection.py

# Test area allocation
python3 scripts/area_allocation.py

# Run complete integrated demo
python3 scripts/integrated_demo.py
```

---

## 🎓 Research Paper References

**Primary Paper:**
- Brust, C., et al. (2021). "DroughtCast: A Machine Learning Forecast of the United States Drought Monitor." *Frontiers in Big Data*, 4, 773478.
- DOI: 10.3389/fdata.2021.773478

**Dataset:**
- Minix, M. (2023). "US Drought Meteorological Data." Kaggle.
- https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data

---

## 📦 Dependencies

Required Python packages:
- `numpy` - Numerical computations
- `scipy` - Statistical functions (expit logistic mapping)
- `pandas` - CSV data handling (optional, for CSV extraction)

Install with:
```bash
pip3 install numpy scipy pandas
```

---

## 🚀 Integration with ROS

The modules are designed to integrate with the ROS-based drone simulation:

1. **Probability Module** → Feed to `area_explorer.py` for risk assessment
2. **Allocation Module** → Determine which drones visit which areas
3. **Verification Module** → Validate sensor readings from `drone_controller.py`
4. **Auditor System** → Deploy additional drones on detected faults

### Expected ROS Integration Points:

```python
# In multi_drone_navigator.py
from drought_probability_model import DroughtProbabilityModel
from area_allocation import DynamicDroneAllocator
from sensor_fault_detection import DroneVerificationSystem

# Initialize
model = DroughtProbabilityModel()
allocator = DynamicDroneAllocator(18, 10)
verifier = DroneVerificationSystem()

# Get probabilities
area_probs = model.get_area_probabilities(10)

# Allocate drones
result = allocator.allocate_drones(areas, drones)

# Verify sensor readings as drones report
verifier.add_measurement(sensor_reading, model_prob)
```

---

## ✅ Implementation Checklist

- [x] Drought probability model (random pool for testing)
- [x] Feature extraction from meteorological data
- [x] Logistic probability mapping
- [x] Sensor fault detection (hypothesis testing)
- [x] Inverse-variance sensor fusion
- [x] Area prioritization by risk
- [x] Dynamic drone allocation algorithm
- [x] Auditor deployment on fault
- [x] Multi-drone verification system
- [x] Integrated end-to-end demo
- [x] Comprehensive documentation
- [ ] ROS node integration (next phase)
- [ ] Real LSTM model training (next phase)
- [ ] Satellite NDVI data integration (next phase)

---

## 🎯 Next Steps

1. **Train LSTM Model:** Replace random probabilities with real ML predictions using Kaggle dataset
2. **ROS Integration:** Connect modules to `multi_drone_navigator.py`
3. **Real Sensors:** Integrate actual sensor models in drone simulation
4. **Satellite Data:** Add MODIS NDVI for vegetation stress calculation
5. **Real-time Dashboard:** Implement RViz visualization of risk levels and drone assignments
6. **Performance Testing:** Benchmark allocation efficiency and fault detection accuracy

---

## 📄 License

MIT License - See LICENSE file in repository

---

## 👨‍💻 Author

**SanyamBK**  
GitHub: [@SanyamBK](https://github.com/SanyamBK)  
Repository: [ROS-BTP-Drone](https://github.com/SanyamBK/ROS-BTP-Drone)

---

## 🙏 Acknowledgments

- Paper authors: Colin Brust et al. for DroughtCast methodology
- Kaggle: For meteorological and drought monitoring datasets
- ROS Community: For excellent robotics framework
- Gazebo: For realistic simulation environment
