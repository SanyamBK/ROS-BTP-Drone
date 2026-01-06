# Multi-Drone Agricultural Monitoring System - Project Updates

## 📅 Latest Updates (December 2025)

### System Configuration Changes

#### Fleet Size Optimization
- **Previous Configuration**: 18 drones for 10 farmland areas
- **Current Configuration**: **11 drones (10 active + 1 backup)** for 5 farmland areas
- **Allocation Strategy**: Fixed 2 drones per farmland area + 1 standby drone
- **Rationale**: Optimized for balanced coverage with redundancy

#### Farmland Area Reorganization
- **Previous**: 10 smaller scattered farmland patches
- **Current**: **5 larger consolidated farmland areas**
  1. **Area 1 (Red)**: Wheat cultivation at (-12, 9) - Farmland 1
  2. **Area 2 (Blue)**: Soybean cultivation at (-2, 11.5) - Farmland 2 (larger area)
  3. **Area 3 (Yellow)**: Maize cultivation at (11, 7.5) - Farmland 3
  4. **Area 4 (Purple)**: Barley cultivation at (-10, 0) - Farmland 4
  5. **Area 5 (Orange)**: Vegetable crops at (1, -1) - Farmland 5

### New Features Implemented

#### 1. Central Communication System with Connection Logging

**Architecture**: Asynchronous 3-Way Handshake Protocol

```
Phase 1: HELLO Broadcast (Central Tower → All Agents)
         ↓
Phase 2: HI Response (Agents → Central Tower, with random delay 0.1-2.0s)
         ↓
Phase 3: ACK Confirmation (Central Tower → Specific Agent)
         ↓
      Connection Established
```

**Key Components**:
- **Central Agent (`central_agent.py`)**: 
  - Static command tower coordinating the entire fleet
  - Broadcasts HELLO beacons every 10 seconds
  - Processes connection requests via asynchronous queue (10Hz)
  - Maintains connection state for all agents
  
- **Drone Communication Manager (`drone_comm.py`)**:
  - Manages communication for all 10 active drones
  - Handles HELLO reception and HI responses
  - Confirms ACK and establishes stable connections
  
- **UGV Communication Manager (`ugv_comm.py`)**:
  - Manages communication for 2 mobile charging UGVs
  - Participates in the same handshake protocol
  - Ensures reliable command/control link

**Connection Logging**:
- All communication events are timestamped and logged to `logs/connection_report.log`
- Events captured:
  - HELLO broadcasts from central tower
  - HI responses from drones and UGVs
  - ACK confirmations
  - Connection establishment confirmations
  - Mission complete notifications

**Log File Format**:
```
[2025-12-29T12:06:51.748529] [CENTRAL_TOWER] Online. Waiting for drone fleet to deploy...
[2025-12-29T12:06:51.748667] [DroneNet] Ready with 10 drones
[2025-12-29T12:07:11.747310] [CENTRAL_TOWER] >>> Broadcasting: HELLO
[2025-12-29T12:07:12.482922] [DRONE_5] Heard HELLO. Sending HI...
[2025-12-29T12:07:12.543325] [DRONE_5] Connection Established! (ACK Received)
[2025-12-29T12:07:13.512058] [UGV_1] Heard HELLO. Sending HI...
[2025-12-29T12:07:13.920156] [UGV_1] Connection Established. (ACK Received)
```

#### 2. Bug Fixes and Code Quality Improvements

**Fixed Issues**:
1. **Indentation Error in `central_agent.py`** (Line 21):
   - Issue: Mission subscription and async queue initialization had incorrect indentation
   - Fix: Properly indented all `__init__` members within the class
   
2. **Indentation Error in `ugv_manager.py`** (Line 72):
   - Issue: `drone_positions` dictionary and related subscriptions were incorrectly indented
   - Fix: Aligned mission_done flag and drone tracking setup within `__init__`
   - Also corrected log prefix formatting from `f"{self.ugv_id}]"` to `f"[{self.ugv_id}]"`
   
3. **Duplicate Code in `ugv_manager.py`**:
   - Removed duplicated turn/velocity logic in `move_towards` method
   
4. **Syntax Validation**:
   - All modified scripts pass `python3 -m py_compile` without errors

### Log Files Overview

The system now generates comprehensive log files in `src/multi_drone_sim/logs/`:

#### 1. **connection_report.log** (NEW)
- **Purpose**: Records all communication events between central tower and agents
- **Contents**:
  - Handshake initialization and completion
  - HELLO broadcast timestamps
  - HI response times (with agent IDs)
  - ACK confirmations
  - Connection establishment events
  - Mission shutdown notifications
- **Use Cases**:
  - Communication reliability analysis
  - Network latency profiling
  - Agent responsiveness monitoring
  - Debugging connection issues

#### 2. **drought_allocation.log**
- **Purpose**: Documents initial drone allocation based on drought risk
- **Contents**:
  - LSTM model predictions for each area
  - Risk scores (0.0 - 1.0)
  - Drone assignment strategy
  - Priority-based allocation decisions
  - Backup drone designation
- **Use Cases**:
  - Verifying allocation algorithm
  - Risk assessment validation
  - Mission planning review

#### 3. **mission_summary.log**
- **Purpose**: Comprehensive mission report with all key metrics
- **Contents**:
  - Total mission duration
  - Per-drone coverage statistics
  - Waypoint completion rates
  - Battery consumption data
  - Risk assessment accuracy (model vs. onboard sensors)
  - Boundary correction counts
  - UGV docking events
  - Final mission status
- **Use Cases**:
  - Performance analysis
  - System optimization
  - Research data collection
  - Mission debriefing

### System Performance Metrics (Latest Run)

**Mission Completion**: ✅ Successful
- **Total Duration**: ~160 seconds (from spawn to mission complete)
- **Active Explorers**: 10 drones
- **Backup Drones**: 1 drone
- **Areas Covered**: 5 farmland regions (100% coverage)
- **Connection Success Rate**: 100% (all drones + 2 UGVs connected)

**Drone Performance**:
| Drone ID | Area | Risk Level | Waypoints | Completion | Risk Error |
|----------|------|------------|-----------|------------|------------|
| 0 | Area 1 (Red) | 55% | 7/7 | 100% | +4.63% |
| 1 | Area 1 (Red) | 55% | 7/7 | 100% | +3.96% |
| 2 | Area 2 (Blue) | 65% | 8/8 | 100% | -3.32% |
| 3 | Area 2 (Blue) | 65% | 8/8 | 100% | +2.44% |
| 4 | Area 3 (Yellow) | 22% | 7/7 | 100% | -2.67% |
| 5 | Area 3 (Yellow) | 22% | 7/7 | 100% | -2.29% |
| 6 | Area 4 (Purple) | 78% | 7/7 | 100% | -9.00% |
| 7 | Area 4 (Purple) | 78% | 7/7 | 100% | +4.45% |
| 8 | Area 5 (Orange) | 75% | 8/8 | 100% | -10.11% |
| 9 | Area 5 (Orange) | 75% | 8/8 | 100% | -3.99% |
| 10 | Backup | N/A | 0/0 | Standby | 0.00% |

**UGV Performance**:
- **UGV 1**: Active patrol and charging (docked drones 1, 6)
- **UGV 2**: Active patrol and charging (docked drone 4)
- **Charging Events**: 3 successful dockings
- **Patrol Route**: Dynamic response to low-battery alerts

**Swarm Localization**:
- **Active Localizers**: 3 drones (0, 1, 2)
- **Belief Uncertainty**: Tracked via covariance trace (tr(Σ))
- **UWB Simulation**: 18 virtual agents
- **Position Updates**: 10Hz frequency

### Technical Architecture Updates

#### Communication Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Central Tower                           │
│  ┌──────────────────────────────────────────────────┐      │
│  │  central_agent.py                                 │      │
│  │  • Broadcasts HELLO every 10s                     │      │
│  │  • Processes HI via async queue (10Hz)           │      │
│  │  • Sends ACK confirmations                        │      │
│  │  • Logs all events to connection_report.log      │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          ↕ /central/comm
                          ↕ /comm/agents
┌─────────────────────────────────────────────────────────────┐
│                     Agent Network                            │
│  ┌────────────────────┐  ┌──────────────────────┐          │
│  │ drone_comm.py      │  │ ugv_comm.py           │          │
│  │ • 10 virtual drones│  │ • 2 virtual UGVs      │          │
│  │ • Responds to HELLO│  │ • Responds to HELLO   │          │
│  │ • Sends HI (w/delay)│  │ • Sends HI (w/delay) │          │
│  │ • Confirms ACK     │  │ • Confirms ACK        │          │
│  │ • Logs events      │  │ • Logs events         │          │
│  └────────────────────┘  └──────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

#### File Structure Updates

```
src/multi_drone_sim/
├── scripts/
│   ├── central_agent.py         ✅ UPDATED (connection logging added)
│   ├── drone_comm.py             ✅ UPDATED (connection logging added)
│   ├── ugv_comm.py               ✅ UPDATED (connection logging added)
│   ├── ugv_manager.py            ✅ FIXED (indentation errors resolved)
│   ├── area_explorer.py          (manages 10 explorer + 1 backup)
│   ├── spawn_fleet.py            (spawns 11 drones)
│   ├── energy_planner.py         (manages UGV charging)
│   ├── swarm_localization.py    (UWB-based positioning)
│   └── uwb_simulator.py          (simulates UWB ranging)
├── logs/
│   ├── connection_report.log     ✅ NEW (communication events)
│   ├── drought_allocation.log    (risk-based allocation)
│   └── mission_summary.log       (comprehensive mission data)
├── config/
│   └── areas_new_config.yaml     (5 farmland areas configuration)
├── launch/
│   └── multi_drone_sim.launch    (main launch file)
└── worlds/
    └── field_areas.world          (Gazebo world with 5 areas)
```

### Configuration Parameters

#### Launch File Parameters (`multi_drone_sim.launch`)

```xml
<!-- Drone Fleet -->
<param name="/drone_comms/num_drones" value="10"/>

<!-- UGV Configuration -->
<node name="ugv1_manager" pkg="multi_drone_sim" type="ugv_manager.py">
    <param name="namespace" value="ugv1"/>
    <param name="ugv_id" value="UGV 1"/>
    <param name="model_name" value="UGV_Charger"/>
    <param name="start_x" value="-6.0"/>
    <param name="start_y" value="4.0"/>
    <param name="min_separation" value="2.0"/>
    <param name="patrol_waypoints" value="[[-12, 9], [-2, 11.5], [11, 7.5]]"/>
</node>

<node name="ugv2_manager" pkg="multi_drone_sim" type="ugv_manager.py">
    <param name="namespace" value="ugv2"/>
    <param name="ugv_id" value="UGV 2"/>
    <param name="model_name" value="UGV_Charger_2"/>
    <param name="start_x" value="5.0"/>
    <param name="start_y" value="10.0"/>
    <param name="min_separation" value="2.0"/>
    <param name="patrol_waypoints" value="[[11, 7], [-2, 11.5], [-10, 0]]"/>
</node>
```

#### Area Configuration (`areas_new_config.yaml`)

```yaml
areas:
  area_1:
    center_x: -12.0
    center_y: 9.0
    radius: 4.5
    color: "red"
    crop_type: "Wheat"
    name: "Farmland 1"
    
  area_2:
    center_x: -2.0
    center_y: 11.5
    radius: 5.5
    color: "blue"
    crop_type: "Soybean"
    name: "Farmland 2"
    
  area_3:
    center_x: 11.0
    center_y: 7.5
    radius: 4.5
    color: "yellow"
    crop_type: "Maize"
    name: "Farmland 3"
    
  area_4:
    center_x: -10.0
    center_y: 0.0
    radius: 4.5
    color: "purple"
    crop_type: "Barley"
    name: "Farmland 4"
    
  area_5:
    center_x: 1.0
    center_y: -1.0
    radius: 5.5
    color: "orange"
    crop_type: "Vegetables"
    name: "Farmland 5"

allocation:
  drones_per_area: 2
  total_drones: 10
  backup_drones: 1
```

### Development Timeline

- **Initial System (Oct 2025)**: 18 drones, 10 areas, basic communication
- **Optimization Phase (Nov 2025)**: Reduced to 11 drones, 5 consolidated areas
- **Communication Enhancement (Dec 2025)**: 
  - Added central agent coordination
  - Implemented 3-way handshake protocol
  - Added comprehensive connection logging
  - Fixed critical indentation bugs
  - Validated all Python scripts

### Known Issues and Solutions

#### 1. Gazebo "Entity Already Exists" Warning
**Issue**: When rerunning simulation without restarting Gazebo
```
Spawn status: SpawnModel: Failure - entity already exists.
```
**Solution**: Clear Gazebo world or restart `gzserver` between runs
```bash
killall gzserver gzclient
roslaunch multi_drone_sim multi_drone_sim.launch
```

#### 2. NumPy Version Warning
**Warning**: 
```
UserWarning: A NumPy version >=1.19.5 and <1.27.0 is required for this version of SciPy
```
**Solution** (optional): Upgrade NumPy
```bash
pip install --upgrade "numpy>=1.19.5,<1.27.0"
```

#### 3. Context Mismatch in SVGA
**Warning**: `context mismatch in svga_surface_destroy`
**Impact**: Cosmetic only, does not affect simulation
**Note**: Related to Mesa/OpenGL graphics drivers, can be ignored

### Future Enhancements

1. **Enhanced Communication Metrics**:
   - Per-agent connection quality scores
   - Packet loss simulation
   - Latency histogram generation
   - Periodic heartbeat mechanism

2. **Advanced Logging**:
   - Real-time connection dashboard
   - Mission replay functionality
   - Comparative analysis between runs
   - CSV export for data science workflows

3. **System Scalability**:
   - Support for 20+ drones
   - Multi-layer farmland (tiered priority)
   - Weather condition simulation
   - Seasonal crop rotation modeling

4. **AI/ML Integration**:
   - Reinforcement learning for path optimization
   - Predictive maintenance for drones
   - Adaptive risk threshold tuning
   - Multi-agent collaborative learning

### Testing and Validation

#### Unit Tests
- ✅ Python syntax validation (`py_compile`)
- ✅ ROS node startup checks
- ✅ Communication handshake verification
- ✅ Log file generation confirmation

#### Integration Tests
- ✅ Full mission completion (5 areas)
- ✅ 100% connection success rate
- ✅ UGV charging functionality
- ✅ Swarm localization accuracy
- ✅ Graceful shutdown on mission complete

#### Performance Benchmarks
- **Handshake Latency**: ~0.5-2.0 seconds per agent
- **Mission Duration**: ~160 seconds for 5 areas
- **CPU Usage**: ~40-60% (Intel i7, 8 cores)
- **Memory**: ~2.5 GB (Gazebo + ROS nodes)
- **Log File Size**: ~50 KB per mission

### References and Citations

1. Kundu et al., "Multi-Robot Communication-Aware Cooperative Belief Space Planning," IROS 2024
2. Shan et al., "Ultra-Wideband Swarm Ranging," INFOCOM 2021
3. Karapetyan et al., "Coverage Planning with a Mobile Recharging UGV," ICRA 2024
4. Brust et al., "DroughtCast: A Machine Learning Approach for Drought Prediction," 2021

### Contributors

- **System Design**: Autonomous multi-agent coordination
- **Communication Protocol**: Central tower with async handshake
- **Connection Logging**: Comprehensive event tracking
- **Bug Fixes**: Indentation and duplicate code resolution
- **Documentation**: Updated README and project documentation

---

## Quick Start Command

```bash
# Navigate to workspace
cd ~/catkin_ws/src/multi_drone_sim

# Run the exploration mission
bash start_exploration.sh

# Monitor connection logs in real-time
tail -f logs/connection_report.log

# View mission summary after completion
cat logs/mission_summary.log
```

---

**Last Updated**: December 29, 2025  
**System Version**: 2.0 (Optimized with Connection Logging)  
**ROS Distribution**: Noetic  
**Gazebo Version**: 11  
**Python Version**: 3.8+

