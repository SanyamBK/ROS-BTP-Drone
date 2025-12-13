#!/usr/bin/env python3

import rospy
import threading
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
from math import sqrt, atan2, cos, sin
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState

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
        
        # Link to Gazebo
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Params
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0
        self.omega = 0.0
        
        self.omega = 0.0
        
        # State Machine
        self.state = "PATROL"
        # UPDATED PATROL: Tighter loop around farm areas (-12,9) and (-2,11.5)
        # Keeps UGV closer to demand.
        self.patrol_waypoints = [(-12, 5), (-2, 5), (-2, 12), (-12, 12)]
        self.current_wp_idx = 0
        
        # Startup Phase Logic
        self.start_time = rospy.Time.now()
        self.init_duration = rospy.Duration(15.0) # Stay at center for 15s
        
        self.charging_radius = 2.0 # meters
        
        # Pubs/Subs
        self.odom_pub = rospy.Publisher('/ugv/odom', Odometry, queue_size=10)
        self.cmd_sub = rospy.Subscriber('/ugv/cmd_vel', Twist, self.cmd_callback)
        self.charge_pub = rospy.Publisher('/ugv/charging_active', Bool, queue_size=10)
        
        # We need to know where drones are to charge them
        # In a real sim we'd use a service or collision detection.
        # Here we subscribe to drone odoms dynamically.
        self.drone_positions = {}
        self.num_drones = 18
        # Subscribe to drone battery levels
        self.drone_batteries = {}
        for i in range(self.num_drones):
            rospy.Subscriber(f'/drone_{i}/odom', Odometry, self.drone_cb, callback_args=i)
            rospy.Subscriber(f'/drone_{i}/battery', Float32, self.battery_cb, callback_args=i)
            
        self.low_battery_threshold = 30.0 # Percentage
        self.target_drone_id = None
            
        self.rate = rospy.Rate(10)
        self.last_time = rospy.Time.now()
        
        rospy.loginfo("[UGV] Mobile Charging Station Initialized at (0,0)")
        
        # Patrol State
        self.last_cmd_time = rospy.Time.now()
        # waypoints already defined above
        
    def run_patrol_logic(self, dt):
        """Simple square patrol when idle."""
        target = self.patrol_waypoints[self.current_wp_idx]
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = sqrt(dx*dx + dy*dy)
        
        if dist < 0.5:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.patrol_waypoints)
            return
            
        target_yaw = atan2(dy, dx)
        diff_yaw = target_yaw - self.yaw
        # Normalize angle
        while diff_yaw > 3.14159: diff_yaw -= 2*3.14159
        while diff_yaw < -3.14159: diff_yaw += 2*3.14159
        
        # Turn first, then move
        if abs(diff_yaw) > 0.1:
            self.omega = 0.5 * (1 if diff_yaw > 0 else -1)
            self.velocity = 0.0
        else:
            self.omega = 0.0
            self.velocity = 1.0 # 1 m/s patrol speed

    def cmd_callback(self, msg):
        self.velocity = msg.linear.x
        self.omega = msg.angular.z
        self.last_cmd_time = rospy.Time.now()

    def drone_cb(self, msg, drone_id):
        self.drone_positions[drone_id] = msg.pose.pose.position

    def battery_cb(self, msg, drone_id):
        self.drone_batteries[drone_id] = msg.data

    def get_priority_target(self):
        """Find the drone with the lowest battery below threshold."""
        lowest_bat = self.low_battery_threshold
        target_id = None
        
        for drone_id, bat in self.drone_batteries.items():
            if bat < lowest_bat:
                lowest_bat = bat
                target_id = drone_id
                
        return target_id

    def update_physics(self):
        current = rospy.Time.now()
        dt = (current - self.last_time).to_sec()
        self.last_time = current
        
        # Check for idle timeout (auto-patrol)
        # Startup Phase: Keep at center for first 15s
        if (current - self.start_time) < self.init_duration:
            self.velocity = 0.0
            self.omega = 0.0
            self.x = 0.0
            self.y = 0.0
        elif (current - self.last_cmd_time).to_sec() > 1.0:
            # INTERCEPT LOGIC
            priority_drone = self.get_priority_target()
            
            if priority_drone is not None and priority_drone in self.drone_positions:
                # Move towards priority drone
                target_pos = self.drone_positions[priority_drone]
                self.move_towards(target_pos.x, target_pos.y, dt)
                if current.to_sec() % 5.0 < 0.1:
                    rospy.loginfo_throttle(5, f"[UGV] Responding to Low Battery: Drone {priority_drone} ({self.drone_batteries[priority_drone]:.1f}%)")
            else:
                # Default Patrol
                self.run_patrol_logic(dt)
        
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
        odom.pose.pose.orientation.z = sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = cos(self.yaw / 2.0)
        
        self.odom_pub.publish(odom)


        # Force Gazebo to match our calculated position
        self.sync_gazebo_state()

    def sync_gazebo_state(self):
        state = ModelState()
        state.model_name = "UGV_Charger"
        state.pose.position.x = self.x
        state.pose.position.y = self.y
        state.pose.position.z = 0.325
        state.pose.orientation.z = sin(self.yaw / 2.0)
        state.pose.orientation.w = cos(self.yaw / 2.0)
        
        try:
            self.set_state(state)
        except rospy.ServiceException:
            pass
    def move_towards(self, tx, ty, dt):
        dx = tx - self.x
        dy = ty - self.y
        dist = sqrt(dx*dx + dy*dy)
        
        if dist < 1.0:
            self.velocity = 0.0
            self.omega = 0.0
            return

        target_yaw = atan2(dy, dx)
        diff_yaw = target_yaw - self.yaw
        while diff_yaw > 3.14159: diff_yaw -= 2*3.14159
        while diff_yaw < -3.14159: diff_yaw += 2*3.14159
        
        if abs(diff_yaw) > 0.1:
            self.omega = 0.8 * (1 if diff_yaw > 0 else -1)
            self.velocity = 0.2
        else:
            self.omega = 0.0
            self.velocity = 1.5 # MAX SPEED RESPONSE

        if abs(diff_yaw) > 0.1:
            self.omega = 0.8 * (1 if diff_yaw > 0 else -1)
            self.velocity = 0.2
        else:
            self.omega = 0.0
            self.velocity = 1.5 # MAX SPEED RESPONSE

    def check_charging(self):
        """Check if any drone is close enough to charge."""
        ugv_pos = Point(self.x, self.y, 0)
        
        for drone_id, pos in self.drone_positions.items():
            dist = sqrt((pos.x - self.x)**2 + (pos.y - self.y)**2)
            
            if dist < self.charging_radius:
                # Drone is docking!
                rospy.loginfo_throttle(5, f"[UGV] Docking success! Drone {drone_id} is recharging...")
                
                # Signal the energy planner (or any battery monitor) that charging is active
                # We publish to a status topic that the planner can verify
                self.charge_pub.publish(True)

    def wait_for_model(self):
        """Block until the UGV model is actually spawned in Gazebo."""
        rospy.loginfo("[UGV] Waiting for 'UGV_Charger' model to spawn...")
        while not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1.0)
                if "UGV_Charger" in msg.name:
                    rospy.loginfo("[UGV] Model spawned! Starting control.")
                    return
            except rospy.ROSException:
                pass
            rospy.loginfo_throttle(2, "[UGV] Still waiting for model spawn...")

    def run(self):
        self.wait_for_model()
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
