#!/usr/bin/env python3

import rospy
import os
import random
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

    # Coordinates from legacy spawn_drones.launch
    positions = [
        (0.0, -20.0), (0.5, -20.0), (-0.5, -20.0),
        (0.0, -19.5), (0.0, -20.5), (0.3, -19.7),
        (-0.3, -19.7), (0.3, -20.3), (-0.3, -20.3),
        (0.5, -19.5), (-0.5, -19.5), (0.7, -20.7),
        (-0.7, -20.7), (0.0, -19.3), (0.5, -20.7),
        (-0.5, -20.7), (0.0, -21.0), (0.3, -21.0)
    ]
    
    # z_height = 0.2 (Old fixed height)

    for i, (x, y) in enumerate(positions):
        drone_name = f"drone_{i}"
        rospy.loginfo(f"Spawning {drone_name} at ({x}, {y})...")
        
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        # Adjust flight altitude to vary between 3.0m and 3.5m
        initial_pose.position.z = random.uniform(3.0, 3.5)
        
        try:
            spawn_model(drone_name, model_xml, f"drone_{i}", initial_pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn failed for {drone_name}: {e}")
            
        rospy.sleep(0.5) # 500ms delay to let Gazebo breathe

if __name__ == '__main__':
    spawn_fleet()
