#!/usr/bin/env python3

import rospy
import os
import random
import math
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

def spawn_fleet():
    rospy.init_node('spawn_fleet_manager')
    
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
    # Path to SDF
    model_path = os.path.expanduser('~/catkin_ws/src/multi_drone_sim/models/quadcopter/model.sdf')
    with open(model_path, 'r') as f:
        model_xml = f.read()



    import yaml
    
    # Load config to get reserve count
    config_path = os.path.expanduser('~/catkin_ws/src/multi_drone_sim/config/areas.yaml')
    try:
        with open(config_path, 'r') as cfg:
            config = yaml.safe_load(cfg)
        num_drones = config.get('num_drones', 11)
        reserve_drones = config.get('allocation', {}).get('reserve_drones', 2)
    except Exception as e:
        rospy.logerr(f"Failed to load config for spawning: {e}")
        num_drones = 11
        reserve_drones = 2

    explorers = max(1, num_drones - reserve_drones)

    # Spawning drones
    center_x = 0.0
    center_y = 0.0   # Centered on farms
    radius = 16.0    # Increased radius to stay completely outside the 21.5x21.5m setup
    
    # Generate num_drones evenly spaced positions
    circle_positions = []
    for i in range(num_drones):
        angle = (2 * math.pi * i) / num_drones
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        circle_positions.append((x, y))
        
    # We want the reserve drones (the LAST 'reserve_drones' IDs) to be spread across.
    # We select evenly spaced geometric indices for the reserve drones.
    reserve_indices = [int(num_drones * i / reserve_drones) for i in range(reserve_drones)] if reserve_drones > 0 else []
    explorer_indices = [i for i in range(num_drones) if i not in reserve_indices]
    
    positions = [None] * num_drones
    
    # First, explorers get the explorer indices
    for i, circle_idx in enumerate(explorer_indices):
        positions[i] = circle_positions[circle_idx]
        
    # Then, reserves get the reserve indices
    for i, circle_idx in enumerate(reserve_indices):
        positions[explorers + i] = circle_positions[circle_idx]

    for i, (x, y) in enumerate(positions):
        drone_name = f"drone_{i}"
        
        rospy.loginfo(f"Spawning {drone_name} at ({x:.2f}, {y:.2f})...")
        
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = random.uniform(2.0, 2.5) # Start at height 2.0m - 2.5m
        
        try:
            spawn_model(drone_name, model_xml, f"drone_{i}", initial_pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn failed for {drone_name}: {e}")
            
        rospy.sleep(0.5) # Increased delay to prevent Gazebo dropping random drones

if __name__ == '__main__':
    spawn_fleet()
