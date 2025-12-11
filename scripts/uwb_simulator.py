#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json

class UWBSimulator:
    """
    Simulates Ultra-Wideband (UWB) ranging between drones.
    Paper: "Ultra-Wideband Swarm Ranging" (Shan et al., INFOCOM 2021)
    
    Function:
    1. Subscribes to true positions (Gazebo ground truth /odometry).
    2. Calculates Euclidean distance between all drone pairs.
    3. Adds Gaussian noise to simulate hardware imperfection.
    4. Publishes Sim-UWB ranges to /swarm/uwb_ranges.
    """
    
    def __init__(self):
        rospy.init_node('uwb_simulator', anonymous=True)
        
        # Configuration
        self.num_drones = 18  # Total drones in simulation
        self.noise_std = 0.10 # 10cm standard deviation for UWB noise
        self.positions = {}   # Store latest ground truth: {drone_id: (x, y, z)}
        self.lock = threading.Lock()
        
        # Subscribers
        for i in range(self.num_drones):
            topic = f'/drone_{i}/odom'
            rospy.Subscriber(topic, Odometry, self.odom_callback, callback_args=i)
            
        # Publisher
        # Format: JSON string list of {id_a: 0, id_b: 1, dist: 5.43}
        # This avoids needing custom ROS message definitions for now.
        self.uwb_pub = rospy.Publisher('/swarm/uwb_ranges', String, queue_size=10)
        
        # Main Loop
        self.rate = rospy.Rate(10) # 10 Hz ranging update
        rospy.loginfo("[UWB-Sim] Initialized UWB Simulator for 18 drones.")

    def odom_callback(self, msg, drone_id):
        pos = msg.pose.pose.position
        with self.lock:
            self.positions[drone_id] = np.array([pos.x, pos.y, pos.z])

    def calculate_ranges(self):
        ranges = []
        with self.lock:
            active_ids = list(self.positions.keys())
            
        # Pairwise distance calculation
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id_a = active_ids[i]
                id_b = active_ids[j]
                
                pos_a = self.positions[id_a]
                pos_b = self.positions[id_b]
                
                # True Euclidean distance
                true_dist = np.linalg.norm(pos_a - pos_b)
                
                # Add Noise (Gaussian)
                noise = np.random.normal(0, self.noise_std)
                measured_dist = max(0.0, true_dist + noise)
                
                # Only report if within realistic UWB range (e.g., 50m)
                if measured_dist < 100.0:
                    ranges.append({
                        'a': id_a,
                        'b': id_b,
                        'dist': round(measured_dist, 3)
                    })
        return ranges

    def run(self):
        while not rospy.is_shutdown():
            if len(self.positions) > 1:
                range_data = self.calculate_ranges()
                if range_data:
                    # Serialize to JSON array for broadcast
                    msg = json.dumps(range_data)
                    self.uwb_pub.publish(msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        sim = UWBSimulator()
        sim.run()
    except rospy.ROSInterruptException:
        pass
