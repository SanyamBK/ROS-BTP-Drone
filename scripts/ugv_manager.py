#!/usr/bin/env python3

import rospy
import threading
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
from math import sqrt, atan2, cos, sin

class MobileRechargingUGV:
    """
    Virtual UGV Manager for ICRA 2024 implementation.
    Acts as a mobile charging station.
    
    Capabilities:
    1. Moves based on /ugv/cmd_vel
    2. Publishes /ugv/odom
    3. Recharges drones that are within 'CHARGING_RADIUS'
    """
    
    def __init__(self):
        rospy.init_node('ugv_manager', anonymous=True)
        
        # Params
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0
        self.omega = 0.0
        self.charging_radius = 2.0 # meters
        
        # Pubs/Subs
        self.odom_pub = rospy.Publisher('/ugv/odom', Odometry, queue_size=10)
        self.cmd_sub = rospy.Subscriber('/ugv/cmd_vel', Twist, self.cmd_callback)
        
        # We need to know where drones are to charge them
        # In a real sim we'd use a service or collision detection.
        # Here we subscribe to drone odoms dynamically.
        self.drone_positions = {}
        self.num_drones = 18
        for i in range(self.num_drones):
            rospy.Subscriber(f'/drone_{i}/odom', Odometry, self.drone_cb, callback_args=i)
            # Publisher to "reset" battery (cheat) or simulated charge speed
            # Since Battery class doesn't have a sub for charging yet, we'll implement
            # a 'charging_active' flag publisher soon.
            
        self.rate = rospy.Rate(10)
        self.last_time = rospy.Time.now()
        
        rospy.loginfo("[UGV] Mobile Charging Station Initialized at (0,0)")

    def cmd_callback(self, msg):
        self.velocity = msg.linear.x
        self.omega = msg.angular.z

    def drone_cb(self, msg, drone_id):
        self.drone_positions[drone_id] = msg.pose.pose.position

    def update_physics(self):
        current = rospy.Time.now()
        dt = (current - self.last_time).to_sec()
        self.last_time = current
        
        # Simple differential drive kinematics
        self.x += self.velocity * cos(self.yaw) * dt
        self.y += self.velocity * sin(self.yaw) * dt
        self.yaw += self.omega * dt
        
        # Publish Odom
        odom = Odometry()
        odom.header.stamp = current
        odom.header.frame_id = "world"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        # Quaternion... simplified
        odom.pose.pose.orientation.w = 1.0 # Ignore rotation for visualization simplicity unless needed
        
        self.odom_pub.publish(odom)

    def check_charging(self):
        """Check if any drone is close enough to charge."""
        ugv_pos = Point(self.x, self.y, 0)
        
        for drone_id, pos in self.drone_positions.items():
            dist = sqrt((pos.x - self.x)**2 + (pos.y - self.y)**2)
            
            if dist < self.charging_radius:
                # Drone is docking!
                # In a real impl, we'd send a 'charge' service request.
                pass 
                # rospy.loginfo_throttle(5, f"Charging Drone {drone_id}...")

    def run(self):
        while not rospy.is_shutdown():
            self.update_physics()
            self.check_charging()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        ugv = MobileRechargingUGV()
        ugv.run()
    except rospy.ROSInterruptException:
        pass
