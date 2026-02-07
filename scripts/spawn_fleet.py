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



    # Spawning 11 drones in a circle around the farmland areas
    # Center approx (0, 5), Radius 25m
    positions = []
    num_drones = 11
    center_x = 0.0
    center_y = 5.0   # Centered on farms
    radius = 25.0    # Wide circle around farms
    
    for i in range(num_drones):
        angle = (2 * math.pi * i) / num_drones
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions.append((x, y))

    for i, (x, y) in enumerate(positions):
        drone_name = f"drone_{i}"
        
        rospy.loginfo(f"Spawning {drone_name} at ({x:.2f}, {y:.2f})...")
        
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = random.uniform(1.5, 2.0) # Start at height 1.5m - 2.0m
        
        try:
            spawn_model(drone_name, model_xml, f"drone_{i}", initial_pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn failed for {drone_name}: {e}")
            
        rospy.sleep(0.2) # Reduced delay as they are far apart

if __name__ == '__main__':
    spawn_fleet()
