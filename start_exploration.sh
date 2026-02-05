#!/bin/bash

# Quick start script for multi-drone area exploration

echo "============================================================"
echo "   Multi-Drone Area Exploration - Quick Start"
echo "============================================================"
echo ""
echo "Mission Configuration:"
echo "  • 11 Total Drones"
echo "    - 10 drones assigned to coverage areas (1-5)"
echo "    - 2 drones per farmland area"
echo "    - 1 standby drone for backup/auditing"
echo "    - Will systematically explore entire area"
echo ""
echo "Area Assignments (Fixed Allocation):"
echo "  • Each of the 5 areas gets exactly 2 drones"
echo "  • 1 reserve drone for backup missions"
echo "  • Areas with higher drought risk monitored continuously"
echo ""
echo "  Area 1 (Red Cylinder)       at (-12, 9)   - Wheat"
echo "  Area 2 (Blue Cylinder)      at (-2, 11.5) - Soybean (larger)"
echo "  Area 3 (Yellow Cylinder)    at (11, 7.5)  - Maize"
echo "  Area 4 (Purple Cylinder)    at (-10, 0)   - Barley"
echo "  Area 5 (Orange Cylinder)    at (1, -1)    - Vegetables"
echo ""
echo "  • 2 UGV Mobile Chargers patrol the farmlands"
echo "  • Circular areas allow for flexible drone positioning"
echo "============================================================"
echo ""
echo "Starting exploration mission..."
echo "Press Ctrl+C to stop"
echo ""

# Source the workspace
source $HOME/catkin_ws/devel/setup.bash

# Make Python scripts executable
chmod +x "$(dirname "$0")"/scripts/*.py

# Launch the exploration simulation
roslaunch multi_drone_sim explore_areas.launch
