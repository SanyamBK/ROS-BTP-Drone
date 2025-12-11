# Research Implementation Plan: Phase 2
## Swarm Intelligence & Energy-Aware Coverage

Based on the research directives from **Tanmoy Kundu (Advisor)**, the next phase of development focuses on implementing specific algorithms from the provided literature.

### 1. Foundation (Theoretical Basis)
**Paper:** *Multi-Robot Communication-Aware Cooperative Belief Space Planning with Inconsistent Beliefs: An Action-Consistent Approach* (Kundu et al., IROS 2024)
*   **Status:** ✅ **Architecture Implemented** (Basic Swarm Setup)
*   **Role:** Defines the cooperative planning framework. Using "Action-Consistent" coordination to handle inconsistent beliefs (e.g., sensor mismatch) between drones.

---

### 2. Swarm Ranging Protocol (Primary Goal)
**Paper:** *Ultra-Wideband Swarm Ranging* (Shan et al., INFOCOM 2021)
*   **Objective:** Implement a decentralized, range-based relative localization system using Ultra-Wideband (UWB) logic.
*   **Current State:** Drones rely on absolute GPS (Global `odom` topic).
*   **Implementation Plan:**
    1.  **Simulate UWB Nodes:** Create a ROS node that simulates noisy distance measurements between drones (`range(d_i, d_j)`).
    2.  **Ranging Protocol:** Implement the specific scheduling/ranging protocol from the paper to avoid packet collisions.
    3.  **Localization Solver:** Implement the multilateration algorithm to estimate relative positions purely from these range inputs.
    4.  **Integration:** Replace `odom` dependency in `area_explorer.py` with this relative position estimate.

---

### 3. Energy-Aware Path Planning
**Paper:** *Coverage Planning with a Mobile Recharging UGV and an Energy-Constrained UAV* (Karapetyan, ICRA 2024)
*   **Objective:** Optimize coverage paths considering battery constraints and mobile charging support.
*   **Current State:** Greedy area allocation (`area_allocation.py`) with no battery modeling.
*   **Implementation Plan:**
    1.  **Energy Model:** Add a battery discharge model to d`rone_controller.py` (based on flight time/distance).
    2.  **UGV Simulation:** Introduce a ground vehicle (UGV) entity in Gazebo that acts as a mobile charging station.
    3.  **Planner Logic:** Implement the ICRA 2024 algorithm to coordinate the Drone-UGV rendezvous points.
    4.  **Mission Controller:** Update `explore_areas.launch` to run this coordinated mission instead of independent patrols.

---

### 4. Action Items
- [x] Create `scripts/uwb_simulator.py` (simulating range inputs)
- [x] Create `scripts/swarm_localization.py` (implementing INFOCOM logic)
- [x] Create `scripts/energy_planner.py` (implementing ICRA logic)
