#!/bin/bash

# Quick start script for multi-drone area exploration

echo "============================================================"
echo "   Multi-Drone Area Exploration - Quick Start"
echo "============================================================"
echo ""
echo "Mission Configuration:"
echo "  • 18 Total Drones"
echo "    - All assigned to coverage areas (1-10)"
echo "    - Multiple drones per area for better coverage"
echo "    - Will systematically explore entire area"
echo ""
echo "Area Assignments (Dynamic Allocation):"
echo "  • Each of the 10 areas gets at least 1 drone"
echo "  • Remaining 8 drones distributed round-robin"
echo "  • Areas with higher drought risk get priority"
echo ""
echo "  Area 1 (Red Cylinder)       at (-10, 10)  - Wheat"
echo "  Area 2 (Blue Cylinder)      at (0, 10)    - Soybean (larger, overlaps)"
echo "  Area 3 (Yellow Cylinder)    at (10, 10)   - Maize"
echo "  Area 4 (Purple Cylinder)    at (-10, 0)   - Barley"
echo "  Area 5 (Orange Cylinder)    at (0, 0)     - Vegetables (largest, overlaps)"
echo "  Area 6 (Cyan Cylinder)      at (10, 0)    - Sunflower"
echo "  Area 7 (Magenta Cylinder)   at (-10, -10) - Rice"
echo "  Area 8 (Lime Cylinder)      at (0, -10)   - Cotton (overlaps)"
echo "  Area 9 (Olive Cylinder)     at (10, -10)  - Peanuts"
echo "  Area 10 (Navy Cylinder)     at (0, 0)     - Pulses (center, overlaps)"
echo ""
echo "  • Circular areas allow for flexible drone positioning"
echo "  • Overlapping regions enable collaborative monitoring"
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
