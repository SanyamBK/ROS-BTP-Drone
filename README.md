# 🚁 Multi-Drone Agricultural Monitoring System

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![Gazebo](https://img.shields.io/badge/Gazebo-11-orange.svg)](http://gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent multi-drone simulation system for autonomous agricultural monitoring, with advanced drought risk assessment and adaptive resource allocation capabilities.

## 📋 Table of Contents

- [Changelog](#changelog)
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [What We've Built](#what-weve-built)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Codebase & API Reference](#codebase--api-reference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a sophisticated multi-drone coordination system designed for precision agriculture applications. The system deploys **11 autonomous drones** over a **unified 21.5 m × 21.5 m workspace**, assesses drought risk using historical climate data, dynamically allocates coverage partitions with Voronoi-based splitting, and recovers from mid-flight hardware failures by activating reserve drones.

### Real-World Application

Climate change is increasing the frequency and severity of agricultural droughts. Our system helps farmers and agricultural managers by:
- **Early drought detection** through multi-sensor analysis
- **Full area coverage** with dynamic Voronoi partitioning across the unified field
- **Intelligent resource allocation** with reserve-drone deployment on hardware failure
- **Comprehensive monitoring** with centralized communication coordination

---

## 📝 Changelog

### Session: 18 April 2026 — Portable LSTM Inference & Realistic Swarm Deployment

**Files changed:** `launch/explore_areas.launch`, `scripts/area_explorer.py`, `scripts/drought_probability_model.py`

#### Feature Enhancement: Portable LSTM Inference (Repository Lightweighting)
- **Requirement:** The 2GB Kaggle Meteorological dataset prevented pushing the repository to external hosts seamlessly.
- **Solution:** Extracted a lightweight 90-day sequential slice of the required meteorological fields (`LSTM/reference_sequence.npy`, ~2.9 KB).
- **Implementation:** `drought_probability_model.py` now attempts to natively load the massive CSV first. If omitted for portability, it seamlessly catches the `FileNotFoundError` and defaults to the reference sequence payload. The complex LSTM inference logic now functions identically on any cloned machine out of the box!

#### Feature Enhancement: Realistic P2P Crash Signaling & Delayed Deployment
- **Fix:** Swarm members no longer cheat by instantly deploying reserves upon hardware failures. Failing drones now broadcast a localized peer-to-peer (P2P) ad-hoc SOS signal.
- **Local Adaptation:** Nearby active drones capture this SOS metric and dynamically absorb the dead vehicle's coverage Voronoi-sector natively and instantly.
- **Tower Restocking:** The Central Tower autonomously detects the missing unit through an authentic 15-second heartbeat timeout delay before legitimately triggering a backup drone to launch from the staging pad.

#### Bug Fix: Simulation Nodes Lingering Post-Mission
- Inserted `required="true"` rigidly onto the master `area_explorer` node in `explore_areas.launch`. When the swarm completes 100% coverage and exits the master node, `roslaunch` automatically intercepts the completion and explicitly tears down the entire simulation tree (Gazebo GUI, UGV logic, Comms, etc.).

---

### Session: 5 April 2026 — Reserve Drone & Stop-Condition Fixes

**Files changed:** `scripts/area_explorer.py`, `scripts/algo3_sim.py`

#### Bug Fix — Second reserve drone never entered the field
When two drones died in the same heartbeat window, the Central Tower published two `DEPLOY_RESERVE` commands. Both backups were correctly dequeued, but the **old `BackupDrone.hold_position()` thread kept running** after promotion, continuously publishing zero-velocity `Twist()` commands that fought against the new `DroneExplorer`'s navigation output. The second reserve appeared frozen.

- Added `self.deployed = False` flag to `BackupDrone.__init__`.
- The `hold_position()` `while` loop guard is now `while not rospy.is_shutdown() and not self.deployed`.
- The `DEPLOY_RESERVE` handler sets `bk.deployed = True` before creating the new `DroneExplorer`, cleanly exiting the old thread.
- Also fixed: the promoted reserve's Voronoi position was seeded from `(0, 0)` (field corner). It is now initialised from the backup's **actual current pose** so partitioning is correct from the first step.

#### Bug Fix — Simulation kept looping at 100% coverage instead of stopping
The `algo3_sim.py` `step()` method previously **reset the entire grid** every time coverage reached 99.9% (continuous surveillance loop). This prevented `AreaCoverageController.is_complete()` from ever returning `True`, so the swarm ran forever.

- Removed the grid-reset block in `Algo3Hybrid.step()`; at ≥ 99.9% it now returns `"Coverage Complete"` and stops assigning new targets.
- Merged the two duplicate `if active_areas == 0` stop blocks in `area_explorer.py`'s main loop into a single, clean check: `if total_pct >= 100.0 or active_areas == 0:`.
- `mission_pub.publish(Bool(data=True))` now fires **inside the main loop** the moment coverage completes, so UGVs park and the Central Tower shuts down immediately — not 5 seconds later at the end of `main()`.
- Removed dead/unreachable code (lines after the original `break` that could never execute).

---

### Commit `69c7b75` — 3 April 2026: UGV Deadlocks, Crash Visuals, Dead-Drone Edge Cases

**Files changed:** `models/dead_quadcopter/model.sdf`, `scripts/algo3_sim.py`, `scripts/area_explorer.py`, `scripts/energy_planner.py`, `scripts/ugv_manager.py`

- **UGV deadlock fix**: Removed redundant local intercept logic from `ugv_manager.py` that caused both UGVs to target the same drone simultaneously, triggering oscillation and deadlock.
- **Crash visuals**: Crashed drones are now teleported to `z = −100 m` (underground) to prevent Gazebo mutex-locking errors that occurred when spawning the static `dead_quadcopter` wreckage model while the physics engine still held a reference to the live model.
- **Ghost cone fix**: The FOV cone `Marker` for a dead drone is explicitly deleted from RViz (using `Marker.action = DELETE`) immediately before the teleport, removing phantom visual artefacts.
- **Dead-drone battery broadcast**: Upon drone death, the system immediately publishes a fake `100%` battery reading for that drone so UGVs do not attempt to drive to and charge a wreck.

---

### Commit `f6431ef` — 2 April 2026: Fleet Stagnation & Dynamic Reconfiguration

**Files changed:** `config/areas.yaml`, `scripts/algo3_sim.py`, `scripts/area_explorer.py`, `scripts/central_agent.py`, `scripts/drone_comm.py`, `scripts/spawn_fleet.py`, `worlds/field_areas.world`

- **`waiting_for_charge` stall fix**: Drones that returned to base for charging were stuck in a `waiting_for_charge` state after recharging. The state is now cleared as soon as battery exceeds 90%, allowing immediate mission resumption.
- **Dead-drone communication silence**: `drone_comm.py` now listens to `/swarm/drone_death`; on receiving a dead drone's ID it stops responding to `HELLO` broadcasts, preventing the Central Tower from endlessly retrying dead units.
- **Central Tower retry hardening** (`central_agent.py`): Added `HELLO_RETRY_<ID>` targeted unicast retries 2.2 s after each broadcast. Drones missing two consecutive heartbeat windows (> 15 s) are now classified as dead and a `DEPLOY_RESERVE` command is fired.
- **Unified workspace** (`field_areas.world`, `areas.yaml`): The five scattered farmland circles were merged into a single **21.5 m × 21.5 m `unified_workspace`** rectangle. All drones now cover this shared space with dynamic Voronoi partitioning, eliminating inter-area boundary conflicts.
- **Reserve-aware spawn** (`spawn_fleet.py`): Now reads `reserve_drones` from `areas.yaml`; reserve drones are spawned in a wider outer ring to avoid collisions with the active explorer formation at startup.

---

### Commit `8ed9f43` — 13 March 2026: Scanning Logic Fixed

**Files changed:** `scripts/algo3_sim.py`, `scripts/area_explorer.py`

- Fixed the scanning state machine so drones hold hover during the 1-second scan pause and only mark a waypoint as visited **after** `check_just_finished()` fires, preventing premature target clearance and the resulting "re-visit loop".
- Introduced `just_finished` flag in `DroneExplorer` to separate the scan-complete event from the algo update tick, eliminating the waypoint double-count that inflated coverage percentages.

---


## ✨ Key Features

### 🤖 Autonomous Drone Fleet Management
- **11 autonomous quadcopters** (up to 9 active explorers + 2 backup/reserve)
- **Unified workspace coverage**: all explorers share one 21.5 m × 21.5 m field with dynamic Voronoi partitioning
- **Multi-threaded execution** for parallel operations
- **Collision avoidance** and safe navigation
- **Dynamic role assignment** (Explorer → Reserve → promoted back to Explorer on demand)
- **Hardware failure simulation**: two random drone deaths injected mid-mission with automatic reserve deployment

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
- **Dynamic Vision (Swept Area)**: Coverage is calculated in real-time based on the "swept area" of the drone's moving field-of-view. The camera FOV is physically modeled as a cone with a 0.625 radius-to-height scaling ratio, ensuring the logical coverage perfectly matches the visual footprint in Gazebo.
- **3D Flight Dynamics**: Drones operate at stable staggered altitudes (2.0m - 3.0m) to maintain realistic vertical separation and avoid downwash. Perfect altitude holding is achieved via zero-gravity planar simulation.
- **Waypoint Grid Density**: The system automatically generates a rigorous scanning grid for each circular area:
    - Points are distributed exactly every **1.5 meters**.
    - Waypoints strictly outside the circular defined farm boundary are rejected.
    - 100% of these 1.5m waypoints must be covered by a drone's camera cone to complete the mission.

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
│  5 Consolidated Circular Farmland Areas                      │
│  ├─ Area 1-5 (larger, consolidated regions)                │
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
All 11 drones share **one unified workspace**. The allocator separates explorers from reserves:
- **9 active explorers** cover the unified field under dynamic Voronoi partitioning
- **2 reserve drones** hold position at spawn and are promoted to explorers if a failure is detected
- Voronoi regions re-partition automatically whenever a drone dies or a reserve is activated

**Fleet configuration (9 explorers / 2 reserves for 11 drones):**
```
Active explorers:  drone_0 … drone_8  (Voronoi-partitioned over unified_workspace)
Reserve drones:    drone_9, drone_10  (hold at spawn, promoted on DEPLOY_RESERVE)
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
- Drone allocation decisions (11 drones across 5 areas)
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
- 11 quadcopter drones (spawn position at y=-20)
- 5 consolidated circular farmland areas
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
  min_drones_per_area: 9    # Minimum active explorers
  max_drones_per_area: 11   # Maximum (full fleet)
  reserve_drones: 2         # Held at spawn, deployed on failure
  measurement_noise: 0.15
  idle_measurement_noise: 0.05
  boundary_soft_margin: 0.4
```

### Drone Fleet Configuration

Total configuration: `config/areas.yaml`
```yaml
num_drones: 11              # Total drone fleet size

start_position:
  x: 0.0
  y: -20.0                  # Spawn outside the 21.5×21.5 m field
  z: 3.5                    # Spawn altitude

# Battery parameters are set in area_explorer.py:
#   Battery(capacity_mah=1500)   — 1500 mAh capacity
#   Initial charge: 100% (drones start fully charged)
```

> **Note:** Drones are spawned in a circle of radius **16 m** centred at (0, 0).
> Explorer drones take the first `num_drones − reserve_drones` slots;
> reserve drones are placed at evenly-spaced angular positions around the same ring.

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

### System Specifications

| Parameter | Value |
|-----------|-------|
| Total Drones | 11 |
| Active Explorers | 9 |
| Reserve Drones | 2 |
| Unified Workspace | 21.5 m × 21.5 m |
| Spawn Radius | 16 m (circle centred at origin) |
| Waypoint Grid Spacing | 1.5 m |
| Battery Capacity | 1500 mAh |
| Low-Battery Threshold | 30% (UGV dispatch trigger) |
| Critical-Battery Threshold | 20% (drone returns to base) |
| UGV Charging Dwell | 1 s (then instant 100% charge command) |
| UGV Navigation Grid | 2 m resolution, 50 × 50 Dijkstra grid |
| Failure 1 Trigger | Random between 10 – 25% field coverage |
| Failure 2 Trigger | Random between 50 – 70% field coverage |
| Heartbeat Timeout | 15 s (Central Tower classifies drone as dead) |

## 📖 Codebase & API Reference

For a comprehensive breakdown of the core ROS nodes, Machine Learning inference logic, battery tracking, and individual Python classes, please refer to the fully detailed **[API & Codebase Reference](CODEBASE_REFERENCE.md)** document.

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
│   └── areas.yaml                    # Unified workspace definition, fleet & allocation config
├── documentation/
│   ├── project_documentation.tex     # Full LaTeX technical document
│   └── project_documentation.pdf     # Compiled PDF
├── launch/
│   ├── multi_drone_sim.launch        # Main simulation launcher
│   ├── spawn_drones.launch           # 11 drone spawning
│   └── explore_areas.launch          # Exploration mission
├── logs/
│   ├── connection_report.log         # Heartbeat / handshake events
│   ├── drought_allocation.log        # Risk-based drone allocation
│   └── mission_summary.log           # Complete mission statistics
├── LSTM/
│   ├── lstm_model.pth            # Trained PyTorch Model weights
│   └── model.py                  # LSTM Class Definition
├── models/
│   ├── dead_quadcopter/              # Static wreckage SDF (spawned on drone death)
│   └── quadcopter/                   # Live drone 3D model
│       ├── model.config
│       ├── model.sdf                 # Full model with FOV cone mesh
│       └── model_no_fov.sdf          # Variant without cone (lightweight)
├── scripts/
│   ├── algo3_sim.py                  # Algo 3: GOMWC + LERR + Shake + Voronoi partitioning
│   ├── area_allocation.py            # Legacy risk-based allocator (research reference)
│   ├── area_explorer.py              # Main mission controller & ROS node entry point
│   ├── central_agent.py              # Central Tower: heartbeat, handshake, DEPLOY_RESERVE
│   ├── drone_comm.py                 # Virtual drone comms relay; silences dead drones
│   ├── drone_controller.py           # Low-level PID drone controller
│   ├── drought_probability_model.py  # LSTM inference wrapper
│   ├── energy_planner.py             # Multi-UGV Dijkstra path planner & charging logic
│   ├── sensor_fault_detection.py     # Statistical fault detection
│   ├── spawn_fleet.py                # Gazebo SDF spawner (ring layout, reserve-aware)
│   ├── swarm_localization.py         # UWB-based decentralised ranging (INFOCOM 2021)
│   ├── ugv_comm.py                   # UGV comms relay (mirrors drone_comm for UGVs)
│   ├── ugv_manager.py                # Mobile UGV physics, patrol, proximity charging
│   └── uwb_simulator.py              # Simulated UWB range measurements
├── worlds/
│   └── field_areas.world             # Gazebo world: unified green field + models
├── CMakeLists.txt
├── package.xml
├── start_simulation.sh
├── start_exploration.sh
└── README.md
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
- [x] **Unified Workspace**: Merge scattered areas into single dynamic Voronoi-partitioned field (Done).
- [x] **Hardware Failure Detection & Reserve Deployment**: Two mid-flight failures simulated; reserves auto-deployed via Central Tower heartbeat timeout (Done).
- [x] **Stop at 100% Coverage**: Simulation halts all drones and UGVs the instant full coverage is confirmed (Done).
- [ ] **Hardware Deployment**: Port to Bitcraze Crazyflie 2.1 swarm for real-world field testing.
- [ ] **Live Weather**: Connect to OpenWeatherMap API.
- [ ] Real hardware deployment (DJI, Pixhawk)
- [ ] Web-based dashboard for monitoring
- [ ] Integration with satellite imagery
- [ ] Collaborative SLAM for area mapping

---

⭐ **Star this repository if you find it useful!**

📧 **Questions?** Open an issue or contact the maintainer.

🚁 **Happy Flying!**
