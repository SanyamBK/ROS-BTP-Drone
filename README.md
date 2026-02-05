# 🚁 Multi-Drone Agricultural Monitoring System

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![Gazebo](https://img.shields.io/badge/Gazebo-11-orange.svg)](http://gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent multi-drone simulation system for autonomous agricultural monitoring, with advanced drought risk assessment and adaptive resource allocation capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [What We've Built](#what-weve-built)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a sophisticated multi-drone coordination system designed for precision agriculture applications. The system deploys **11 autonomous drones (10 active + 1 backup)** to monitor **5 consolidated farmland areas**, assess drought risk using historical climate data, and dynamically allocate resources based on real-time sensor measurements.

### Real-World Application

Climate change is increasing the frequency and severity of agricultural droughts. Our system helps farmers and agricultural managers by:
- **Early drought detection** through multi-sensor analysis
- **Full area coverage** with 2 drones per farmland area
- **Intelligent resource allocation** prioritizing high-risk areas
- **Comprehensive monitoring** with centralized communication coordination

### 🆕 Latest Updates (December 2025)
- **Optimized fleet size**: Reduced from 18 to 11 drones for efficiency
- **Consolidated areas**: 5 larger farmland regions (vs. 10 scattered patches)
- **Connection logging**: New `logs/connection_report.log` tracks all communication events
- **Central coordination**: Asynchronous 3-way handshake protocol (HELLO → HI → ACK)
- **Bug fixes**: Resolved indentation errors in `central_agent.py` and `ugv_manager.py`

## ✨ Key Features

### 🤖 Autonomous Drone Fleet Management
- **11 autonomous quadcopters** (10 explorers + 1 backup)
- **Fixed allocation**: 2 drones per farmland area for balanced coverage
- **Multi-threaded execution** for parallel operations
- **Collision avoidance** and safe navigation
- **Dynamic role assignment** (Explorer, Backup)
- **Distributed across 5 farmland areas** for complete coverage

### 📊 Intelligent Drought Risk Assessment
- **LSTM Neural Network** (PyTorch) trained on US Drought Monitor data:
  - Sequence-to-One architecture (90-day lookback)
  - inputs: Precipitation, Soil Moisture (QV2M), Skin Temp, etc.
- **Probabilistic forecasting** (0.0 - 1.0 Risk Score)
- **Fault Tolerance**: Automatic detection of sensor failures using statistical deviation from model predictions.
- **Swarm Ranging (Active)**: Decentralized UWB-based localization from *INFOCOM 2021* fully integrated.
- **Energy-Aware Planning (Active)**: Cooperative recharging with mobile UGV station from *ICRA 2024*. UGV actively intercepts low-battery drones.
- **Auto-Shutdown**: Simulation automatically terminates 5 seconds after all exploration missions are complete, facilitating batch experiments.
- **Centralized Communication**: A static "Central Tower" node coordinates the fleet using a robust **Asynchronous 3-Way Handshake** (Hello → Hi → Queue → Ack). The tower processes connection requests via a 10Hz queue to simulate realistic processing latency.
- **Connection Logging**: All communication events (HELLO broadcasts, HI responses, ACK confirmations) are timestamped and logged to `logs/connection_report.log`
- **Dynamic Vision (Swept Area)**: Coverage is calculated in real-time based on the "swept area" of the drone's moving field-of-view, simulating realistic sensor footprint data collection.
- **3D Flight Dynamics**: Drones operate at variable altitudes (3.0m - 3.5m) to maintain realistic vertical separation and diverse sensor perspectives.

## 📚 Research Foundation
The system's architecture is built upon the following key research papers:
1.  **Foundation**: *Multi-Robot Communication-Aware Cooperative Belief Space Planning* (Kundu et al., IROS 2024).
2.  **Swarm Ranging**: *Ultra-Wideband Swarm Ranging* (Shan et al., INFOCOM 2021).
3.  **Energy Planning**: *Coverage Planning with a Mobile Recharging UGV* (Karapetyan et al., ICRA 2024).

> **Note**: The system now explicitly calculates and logs the **Belief Uncertainty** ($\text{tr}(\Sigma)$) as per the IROS 2024 paper. It uses the **Moore-Penrose Pseudo-Inverse** (`pinv`) to ensure robust estimation even when anchor geometry is rank-deficient (collinear).

## 🛠️ Tech Stack
- **Fallback Mechanism**: Gracefully degrades to heuristic model if model/deps missing
- **Research Basis**: "DroughtCast" (Brust et al., 2021)

### 🎯 Adaptive Resource Allocation
- **Fixed 2-drone allocation** per farmland area for balanced coverage
- **Priority-based deployment** to highest-risk areas
- **1 backup drone** for redundancy and emergency response
- **Real-time monitoring** based on field measurements

### 🔍 Sensor Fusion & Validation
- **Simulated sensor noise** for realistic scenarios
- **Fault detection** identifying malfunctioning sensors
- **Weighted sensor fusion** combining multiple measurements
- **Variance-based weighting** (lower noise = higher trust)

### 📈 Visualization & Monitoring
- **RViz markers** showing drone positions and risk levels
- **Color-coded risk indicators** (red = high, green = low)
- **Real-time status updates** via ROS topics
- **Comprehensive logging** for mission analysis:
  - `connection_report.log` - Communication events and handshakes
  - `drought_allocation.log` - Risk-based drone allocation
  - `mission_summary.log` - Complete mission statistics

## 📡 ROS Communication Architecture

The system relies on a distributed node architecture with specific topics for command, control, and coordination:

| Node Name | Function | Published Topics | Subscribed Topics |
|-----------|----------|------------------|-------------------|
| `central_agent` | Fleet Command Tower | `/central/comm` (String) | `/comm/agents` (String)<br>`/mission_complete` (Bool) |
| `drone_comm_manager` | Drone Comm. Relay (10 drones) | `/comm/agents` (String) | `/central/comm` (String) |
| `ugv_comm_manager` | UGV Comm. Relay (2 UGVs) | `/comm/agents` (String) | `/central/comm` (String) |
| `area_explorer` (x11) | Drone Autonomy | `/drone_{id}/cmd_vel` (Twist)<br>`/drone_{id}/battery` (Float32) | `/drone_{id}/odom` (Odometry)<br>`/drone_{id}/charge_cmd` (Float32) |
| `ugv_manager` (x2) | Mobile Charging Station | `/ugv_{id}/odom` (Odometry)<br>`/ugv_{id}/charging_active` (Bool) | `/drone_{id}/odom` (Odometry)<br>`/drone_{id}/battery` (Float32)<br>`/mission_complete` (Bool) |

### Key Topic Functions
- **`/central/comm`**: Global broadcast channel for the Central Tower (e.g., `HELLO`).
- **`/comm/agents`**: Return channel for Distributed Agents (e.g., `AGENT_HI_DRONE_5`, `AGENT_HI_UGV_1`).
- **`/drone_{id}/odom`**: Local odometry data for each drone (simulated GPS/IMU).
- **`/mission_complete`**: Mission completion signal triggering graceful shutdown.

### Communication Protocol: 3-Way Handshake

```
Central Tower                          Agents (Drones/UGVs)
     |                                          |
     |  ──────── HELLO (broadcast) ──────────> |  (Step 1)
     |                                          |
     |  <──── AGENT_HI_{ID} (with delay) ────  |  (Step 2)
     |         (queued for processing)          |
     |                                          |
     |  ──────── TOWER_ACK_{ID} ─────────────> |  (Step 3)
     |                                          |
     |        Connection Established ✓          |
     |  (logged to connection_report.log)       |
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gazebo Simulation                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Drone 0  │  │ Drone 1  │  │  ...     │  │ Drone 10 │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  10 Scattered Circular Farmland Areas                        │
│  ├─ Area 1-10 (varying radii, realistic scatter)           │
│  └─ Some overlapping regions for collaborative coverage    │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   ROS Network │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ area_explorer  │  │multi_drone  │  │ drone_controller│
│     .py        │  │ _navigator  │  │      .py        │
│                │  │    .py      │  │                 │
│ • Risk Model   │  │ • Navigator │  │ • Low-level    │
│ • Allocation   │  │ • Sensors   │  │   Control      │
│ • Mission      │  │ • Fusion    │  │ • PID Control  │
│   Planning     │  │ • Markers   │  │ • Odometry     │
└────────────────┘  └─────────────┘  └─────────────────┘
```

## 🔨 What We've Built

### 1. **Drought Risk Analysis Engine (LSTM)**
We replaced the initial heuristic model with a fully trained **Long Short-Term Memory (LSTM)** Neural Network:
- **Architecture**: PyTorch-based LSTM with 2 hidden layers
- **Input**: 90-day history of 6 meteorological features
- **Performance**: Capable of predicting USDM scores with high accuracy
- **Integration**: Runs inference directly within the ROS simulation loop

*The legacy heuristic model is preserved as a fallback.*

The model uses logistic mapping to provide probabilistic forecasts between 5% and 95% confidence levels.

### 2. **Intelligent Drone Allocation System**
Our allocation algorithm ensures complete coverage with dynamic distribution:
- **Coverage guarantee**: Every area gets at least 1 drone
- **Distributed scaling**: Remaining drones added via round-robin
- **Risk prioritization**: Higher-risk areas get first picks
- **Real-world layout**: Scattered positioning (not grid-based)

**Example Allocation for 18 Drones / 10 Areas:**
```
Areas 1-10: 1 drone each (10 drones)
Areas 1-10: Round-robin distribution of 8 remaining drones
Result: 1-3 drones per area depending on risk assessment
```

### 3. **Active UGV Charging System**
We implemented an autonomous Mobile Charging Station (UGV) that:
- **Patrols** the farm perimeter when idle.
- **Intercepts** drones with critical battery levels (<30%).
- **Recharges** drones via a command-based handshake (`/charge_cmd`) to prevent infinite looping.
- **Visual Feedback**: The UGV is fully visualized in Gazebo, distinct from the drones.

### 3. **Realistic Farmland Layout**
Farmland areas are now:
- **Circular instead of square** for natural field representation
- **Scattered with varied spacing** (not rigid 3×3 grid)
- **Strategically overlapping** in some regions for collaborative monitoring
- **Positioned to represent real-world farm parcels**

### 4. **Multi-Drone Navigation System**
Each drone features:
- **Waypoint navigation** with PID control
- **Area scanning** with systematic coverage patterns
- **Boundary enforcement** preventing out-of-bounds flight
- **Real-time odometry** tracking position and velocity
- **Coordinated flight** with other drones in same area

### 5. **Comprehensive Mission Logging**
Detailed logs capture:
- Pre-mission risk assessments
- Drone allocation decisions (18 drones across 10 areas)
- Real-time sensor measurements
- Fault detection events
- Corrected risk estimates
- Mission outcomes

## 🧠 Drought Monitoring Implementation
Based on **"DroughtCast: A Machine Learning Forecast of the United States Drought Monitor"** (Brust et al., 2021).

### Modules
1. **Drought Probability Model**: Estimates risk using Rainfall Deficit, Soil Moisture, Vegetation Stress, etc.
2. **Sensor Fault Detection**: Uses statistical hypothesis testing to identify malfunctioning sensors.
3. **Sensor Fusion**: Combines multiple readings using inverse-variance weighting.
4. **Dynamic Allocation**: Prioritizes high-risk areas and deploys auditors to verify faults.

### Python Examples
**Generate Risk Rankings:**
```python
from area_allocation import AreaPrioritizer, Area
areas = [Area("wheat", 0.85), Area("corn", 0.45)]
ranked = AreaPrioritizer().rank_areas_by_risk(areas)
# Output: wheat (85%), corn (45%)
```

**Detect Faulty Sensors:**
```python
from sensor_fault_detection import SensorFaultDetector
detector = SensorFaultDetector()
is_faulty, _, _ = detector.detect_fault(model_prob=0.65, sensor_prob=0.25, noise=0.05)
# Output: True (deviation too high)
```

## 📦 Prerequisites

### Required Software
- **Ubuntu 20.04** (or compatible Linux distribution)
- **ROS Noetic** - Robot Operating System
- **Gazebo 11** - 3D robot simulator
- **Python 3.8+** - Programming language
- **catkin** - ROS build system

### Required ROS Packages
```bash
sudo apt-get install ros-noetic-gazebo-ros-pkgs
sudo apt-get install ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-hector-gazebo-plugins
sudo apt-get install ros-noetic-teleop-twist-keyboard
sudo apt-get install ros-noetic-teleop-twist-keyboard
sudo apt-get install ros-noetic-teleop-twist-keyboard
sudo apt-get install ros-noetic-rviz

```

### Python Dependencies
```bash
pip3 install pyyaml numpy
# Required for LSTM Model:
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip3 install scipy
```

## 🚀 Installation

### 1. Clone the Repository
```bash
cd ~/catkin_ws/src
git clone https://github.com/SanyamBK/ROS-BTP-Drone.git multi_drone_sim
```

### 2. Build the Workspace
```bash
cd ~/catkin_ws
catkin_make
```

### 3. Source the Workspace
```bash
source ~/catkin_ws/devel/setup.bash
```

*Add this line to your `~/.bashrc` for automatic sourcing:*
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

### 4. Make Scripts Executable
```bash
cd ~/catkin_ws/src/multi_drone_sim
chmod +x scripts/*.py
chmod +x *.sh
```

## 🎮 Usage

### Quick Start (Recommended)

#### Option 1: Using Launch Scripts
```bash
# Terminal 1: Start exploration mission
cd ~/catkin_ws/src/multi_drone_sim
./start_exploration.sh

# Alternatively, for basic simulation
./start_simulation.sh
```

#### Option 2: Manual Launch
```bash
# Terminal 1: Launch Gazebo simulation
roslaunch multi_drone_sim multi_drone_sim.launch

# Terminal 2: Start exploration
roslaunch multi_drone_sim explore_areas.launch
```

### Visualization

#### View in Gazebo
The Gazebo window shows:
- 18 quadcopter drones (spawn position at y=-20)
- 10 scattered circular farmland areas
- Real-time drone movements and area coverage

- Overlapping regions for collaborative monitoring
- **Mobile UGV Charger**: A ground vehicle patrolling and servicing drones.
- **Central Command Tower**: A centralized static structure visualizing the coordination hub.


#### View in RViz (Optional)
```bash
rosrun rviz rviz
```
Add markers for risk visualization:
- **Topic**: `/risk_markers`
- **Type**: `MarkerArray`

### Monitor Mission Progress
```bash
# Watch ROS logs
rostopic echo /rosout

# Monitor drone positions
rostopic echo /drone_0/odom

# View allocation log
cat ~/catkin_ws/src/multi_drone_sim/logs/drought_allocation.log
```

## ⚙️ Configuration

### Area Configuration (`config/areas.yaml`)

Define farmland areas with historical drought data and scattered positions:

```yaml
areas:
  area_1:
    name: "Farmland 1"
    crop: "Wheat"
    x: -12.0      # Scattered X coordinate
    y: 9.0        # Scattered Y coordinate
    z: 2.0        # Altitude
    color: "red"
    size: 10.0    # Diameter in meters
    drought_history:
      - year: 2025
        rainfall_deficit: 0.62      # 0-1 scale
        soil_moisture_index: 0.33   # 0-1 scale
        veg_health_index: 0.41      # 0-1 scale
        heatwave_days: 18
        drought_declared: true
```

### Drone Allocation Parameters

Modify in `config/areas.yaml`:
```yaml
allocation:
  min_drones_per_area: 1    # Minimum drones per area (ensures coverage)
  max_drones_per_area: 3    # Maximum drones per area
  reserve_drones: 0         # All drones assigned to areas
  measurement_noise: 0.15
  idle_measurement_noise: 0.05
  boundary_soft_margin: 0.4
```

### Drone Fleet Configuration

Total configuration: `config/areas.yaml`
```yaml
num_drones: 18              # Total drone fleet size

start_position:
  x: 0.0
  y: -20.0                  # Spawn position (away from farmlands)
  z: 2.0

battery:
  capacity: 800             # mAh (Reduced for simulation speed)
  start_charge_min: 40      # %
  start_charge_max: 90      # %
```

## 🔬 Technical Details

### Drought Risk Model (LSTM)

**Input Tensor (Sequence):** `(1, 90, 6)`
**Features:**
1.  **PRECTOT**: Precipitation
2.  **QV2M**: Specific Humidity (Soil Proxy)
3.  **T2M_MAX**: Max Temperature
4.  **T2M_MIN**: Min Temperature
5.  **TS**: Earth Skin Temperature (Veg Stress Proxy)
6.  **PS**: Surface Pressure

**Output:** Single float `0.0 - 1.0` representing normalized drought risk.

### Navigation Algorithm

**Waypoint Controller:**
```python
1. Calculate distance and angle to target
2. Rotate to face target
3. Move forward with speed proportional to distance
4. Decelerate near target (threshold: 0.5m)
5. Hover when reached (threshold: 0.3m)
```

**Exploration Pattern:**
```python
1. Divide circular area into grid cells
2. Generate waypoints covering each cell
3. Visit waypoints in sequence
4. Take sensor measurements at each point
5. Aggregate measurements for area assessment
```

### Sensor Fusion Algorithm

**Variance-weighted fusion:**
```python
σ²ᵢ = sensor_i_variance
wᵢ = 1 / σ²ᵢ                    # Weight inversely proportional to variance
μ_fused = Σ(wᵢ × μᵢ) / Σ(wᵢ)   # Weighted average
σ²_fused = 1 / Σ(wᵢ)            # Combined variance
```

### UGV Path Planning
- **Algorithm**: Dijkstra's Algorithm (Grid-based)
- **Resolution**: 2.0m grid cells
- **Cost Function**: Uniform cost (shortest path)
- **Fallback**: Direct P-Control if no path found

## 📊 System Specifications

### Fleet Composition
- **Total Drones**: 18 autonomous quadcopters
- **Allocation Strategy**: Min 1, Max 3 per area
- **Coverage**: 10 scattered circular farmland areas
- **Spawn Position**: (0, -20, 2) - away from active zones

### Farmland Layout
- **Area Count**: 10 circular patches
- **Layout**: Scattered, realistic distribution
- **Radii**: Mix of 5-6 unit radii for varied area sizes
- **Overlap**: Strategic overlapping in select regions

### Performance Metrics
- **Mission Success Rate**: 100% (all drones reach assigned areas)
- **Position Accuracy**: ±0.2m (within area boundaries)
- **Execution Time**: ~60 seconds for full 18-drone deployment
- **Coverage Time**: ~120-180 seconds per area exploration

## 🎯 Use Cases

1. **Agricultural Monitoring**: Deploy drones to assess crop health and irrigation needs
2. **Drought Early Warning**: Identify high-risk areas before severe impact
3. **Resource Optimization**: Allocate water/irrigation resources efficiently
4. **Multi-Agent Coordination**: Test fleet management and cooperative control
5. **Research Platform**: Test multi-agent coordination algorithms

## 🛠️ Troubleshooting

### Gazebo Won't Start
```bash
killall gzserver gzclient
roslaunch multi_drone_sim multi_drone_sim.launch
```

### Drones Not Reaching All Areas
Check `areas.yaml` coordinates match `worlds/field_areas.world` positions. Areas should be scattered, not in rigid grid.

### Python Script Errors
Ensure scripts are executable:
```bash
chmod +x scripts/*.py
```

### ROS Package Not Found
Source the workspace:
```bash
source ~/catkin_ws/devel/setup.bash
```



## 📝 Project Structure

```
multi_drone_sim/
├── config/
│   └── areas.yaml                    # 10 area definitions, 18 drone config
├── include/
│   └── multi_drone_sim/              # C++ headers (if needed)
├── launch/
│   ├── multi_drone_sim.launch        # Main simulation launcher
│   ├── spawn_drones.launch           # 18 drone spawning
│   └── explore_areas.launch          # Exploration mission
├── logs/
│   └── drought_allocation.log        # Mission reports
├── LSTM/
│   ├── lstm_model.pth            # Trained PyTorch Model weights
│   └── model.py                  # LSTM Class Definition
├── LSTM_GUIDE.md                 # Documentation for Model Training
├── LSTM_TRAINING_README.md       # Original Training Notes
├── models/
│   └── quadcopter/                   # Drone 3D model
│       ├── model.config
│       └── model.sdf
├── scripts/
│   ├── area_explorer.py              # Risk model & allocation
│   ├── drought_probability_model.py  # Inference Engine (LSTM + Fallback)
│   ├── multi_drone_navigator.py      # Navigation & sensors
│   └── drone_controller.py           # Low-level control
├── src/                              # C++ source files (if needed)
├── worlds/
│   └── field_areas.world             # Gazebo world (10 circular areas)
├── CMakeLists.txt                    # Build configuration
├── package.xml                       # ROS package manifest
├── start_simulation.sh               # Quick-start script
├── start_exploration.sh              # Exploration launcher
└── README.md                         # This file
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**SanyamBK**
- GitHub: [@SanyamBK](https://github.com/SanyamBK)
- Repository: [ROS-BTP-Drone](https://github.com/SanyamBK/ROS-BTP-Drone)

## 🙏 Acknowledgments

- **ROS Community** for the excellent robotics framework
- **Gazebo** for the realistic simulation environment
- **Hector Quadrotor** for drone control plugins
- Agricultural monitoring research inspiring this project

## 📚 References

- [ROS Documentation](http://wiki.ros.org/)
- [Gazebo Tutorials](http://gazebosim.org/tutorials)
- [Multi-Agent Systems in Agriculture](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/precision-agriculture)
- [Drought Monitoring Techniques](https://www.drought.gov/topics/monitoring)

## 🔮 Future Enhancements
- [x] **LSTM Integration**: Replace heuristic model with Deep Learning (Done).
- [x] **Swarm Ranging**: Implement decentralized UWB protocol (Done).
- [x] **Energy Planning**: Implement UGV rendezvous reasoning (Done).
- [ ] **Hardware Deployment**: Port to Bitcraze Crazyflie 2.1 swarm for real-world field testing.
- [ ] **Live Weather**: Connect to OpenWeatherMap API.
- [ ] Real hardware deployment (DJI, Pixhawk)
- [ ] Web-based dashboard for monitoring
- [ ] Integration with satellite imagery
- [ ] Collaborative SLAM for area mapping
- [ ] Dynamic task reassignment mid-mission

---

⭐ **Star this repository if you find it useful!**

📧 **Questions?** Open an issue or contact the maintainer.

🚁 **Happy Flying!**
