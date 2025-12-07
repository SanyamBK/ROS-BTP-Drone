#!/bin/bash

# Quick start script for multi-drone simulation

echo "========================================"
echo "Multi-Drone Simulation - Quick Start"
echo "========================================"
echo ""

# Source the workspace
source $HOME/catkin_ws/devel/setup.bash

echo "Starting simulation..."
echo "- 18 drones will spawn at the starting position"
echo "- They will automatically navigate to 10 colored areas"
echo "- Watch them in Gazebo!"
echo ""
echo "Press Ctrl+C to stop the simulation"
echo ""

# Launch the simulation
roslaunch multi_drone_sim multi_drone_sim.launch
