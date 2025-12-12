#!/usr/bin/env python3

import rospy
import yaml
import os
import threading
import random
from datetime import datetime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import sqrt, atan2, exp, pi, cos, sin
from tf.transformations import euler_from_quaternion
import sys
# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from drought_probability_model import DroughtProbabilityModel
from area_allocation import Area, Drone, DroneRole, DynamicDroneAllocator


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


class MissionAggregator:
    """Thread-safe collector for per-drone mission summaries."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results = []

    def add_result(self, result):
        with self._lock:
            self._results.append(result)

    def get_results(self):
        with self._lock:
            return list(self._results)


def analyze_drought_risk(area_name, area_config):
    """Calculate drought probability and supporting metrics for an area."""
    # Initialize drought model
    model = DroughtProbabilityModel()
    
    # Restore history for the return value
    history = area_config.get('drought_history', [])
    
    # 1. Try LSTM Prediction (highest priority)
    probability = None
    data_path = "/home/ros/catkin_ws/src/multi_drone_sim/us-drought-meteorological-data/versions/5/train_timeseries/train_timeseries.csv"
    
    if os.path.exists(data_path):
        probability = model.predict_from_csv(data_path)
        if probability is not None:
             rospy.loginfo(f"[{area_name}] Using Trained LSTM Model -> Risk: {probability:.1%}")

    # 2. Fallback to Random Pool if LSTM failed (missing torch or model)
    if probability is None:
        probability = model.get_random_probability()

    # Extract features for reporting (works regardless of LSTM)
    features = {}
    if os.path.exists(data_path):
        try:
            features = model.extract_features_from_csv(data_path)
        except Exception as e:
            rospy.logwarn(f"Failed to extract features: {e}")
    
    return {
        'probability': probability,
        'avg_rainfall_deficit': features.get('rain_deficit', 0.0),
        'avg_soil_moisture_deficit': features.get('soil_deficit', 0.0),
        'avg_vegetation_stress': features.get('veg_stress', 0.0),
        'avg_heatwave_intensity': features.get('heat_index', 0.0),
        'historical_drought_rate': features.get('drought_freq', 0.0),
        'trend_factor': features.get('trend', 0.0),
        'years': history
    }


# Local allocate_drones removed. Using AreaPrioritizer from area_allocation.py


def build_allocation_report(log_path, areas_cfg, area_profiles, allocation, full_plan, mission_results=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append('MULTI-DRONE FARMLAND DROUGHT ALLOCATION REPORT')
    lines.append(f'Generated (UTC): {timestamp}')
    lines.append('')
    lines.append('Drought probability model:')
    lines.append('  * Weighted rainfall deficit, soil moisture deficit, vegetation stress, and heatwave intensity')
    lines.append('  * Recent trend bonus captures worsening climatic signals')
    lines.append('  * Logistic mapping keeps probabilities within 5%-95% confidence range')
    lines.append('')
    lines.append('Area drought risk summary:')
    lines.append('Area   Farm Name           P(drought)  RainDef  SoilDef  VegStress  HeatIdx  DroughtFreq  Trend')
    lines.append('-----  ------------------  ----------  -------  -------  ---------  -------  -----------  -----')

    ordered_areas = sorted(areas_cfg.keys(), key=lambda name: area_profiles[name]['probability'], reverse=True)
    for name in ordered_areas:
        profile = area_profiles[name]
        area_cfg = areas_cfg[name]
        farm_name = area_cfg.get('name', name)
        lines.append(
            f"{name:<5}  {farm_name:<18}  {profile['probability']*100:>9.2f}%  "
            f"{profile['avg_rainfall_deficit']:>6.2f}  {profile['avg_soil_moisture_deficit']:>6.2f}  "
            f"{profile['avg_vegetation_stress']:>7.2f}  {profile['avg_heatwave_intensity']:>6.2f}  "
            f"{profile['historical_drought_rate']:>10.2f}  {profile['trend_factor']:>5.2f}"
        )

    lines.append('')
    lines.append('Drone-to-area allocation:')
    lines.append('Drone  Role       Assignment (segment)              Risk   Notes')
    lines.append('-----  ---------  -------------------------------  ------  ---------------------------')

    for idx, plan in enumerate(full_plan):
        if plan['role'] == 'explorer':
            area_name = plan['area']
            area_cfg = areas_cfg[area_name]
            farm_name = area_cfg.get('name', area_name)
            segment = f"{plan['group_index'] + 1}/{plan['group_size']}"
            risk = area_profiles[area_name]['probability'] * 100.0
            notes = f"{farm_name} ({area_cfg.get('color', 'n/a')})"
            lines.append(f"{idx:<5}  explorer  {area_name:<7} {farm_name:<20} {segment:<9}  {risk:>6.1f}%  {notes}")
        else:
            lines.append(f"{idx:<5}  backup    staging-area                 n/a       ---    Holding position")

    mission_results = mission_results or []
    if mission_results:
        lines.append('')
        lines.append('Mission risk observations:')
        lines.append('Drone  Farm Name           Actual  Onboard  Error   Bounds  Final (x,y)     Status   Notes')
        lines.append('-----  ------------------  -------  -------  -------  ------  --------------  -------  ---------------------------')

        for result in sorted(mission_results, key=lambda entry: entry['drone_id']):
            actual = result.get('actual_probability')
            onboard = result.get('measured_probability')
            error_pct = result.get('error_pct')
            boundary_events = result.get('boundary_events', 0)
            final_pos = result.get('final_position', '--')
            status = result.get('status', 'n/a')
            notes = result.get('notes', '')

            actual_str = f"{actual * 100:6.1f}%" if actual is not None else '  --  '
            onboard_str = f"{onboard * 100:6.1f}%" if onboard is not None else '  --  '
            error_str = f"{error_pct:+6.2f}%" if error_pct is not None else '  --  '

            lines.append(
                f"{result['drone_id']:<5}  {result.get('farm_name', 'n/a'):<18}  "
                f"{actual_str:<7}  {onboard_str:<7}  {error_str:<7}  "
                f"{boundary_events:<6}  {final_pos:<14}  {status:<7}  {notes}"
            )

    lines.append('')
    with open(log_path, 'w') as log_file:
        log_file.write('\n'.join(lines))

class DroneExplorer:
    """Explorer drone that patrols a specific area"""

    def __init__(self, drone_id, area_name, area_config, exploration_config, start_pos,
                 result_aggregator, measurement_noise=0.1, boundary_soft_margin=0.3,
                 group_index=0, group_size=1, drought_probability=None):
        self.drone_id = drone_id
        self.area_name = area_name
        self.area_config = area_config
        self.exploration_config = exploration_config
        self.start_pos = start_pos
        self.current_pose = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.exploration_complete = False
        self.group_index = group_index
        self.group_size = max(1, group_size)
        self.result_aggregator = result_aggregator
        self.measurement_noise = abs(measurement_noise)
        self.boundary_soft_margin = max(0.05, boundary_soft_margin)
        self.farm_name = area_config.get('name', area_name)

        self.actual_probability = drought_probability if drought_probability is not None else 0.0
        self.measured_probability = clamp(
            self.actual_probability * (1.0 + random.uniform(-self.measurement_noise, self.measurement_noise))
        )
        self.risk_error_pct = (self.measured_probability - self.actual_probability) * 100.0
        self.boundary_events = 0
        self.notes = []
        self.result_recorded = False

        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(f'/drone_{drone_id}/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber(f'/drone_{drone_id}/odom', Odometry, self.odom_callback)

        # Generate waypoints for area exploration
        self.generate_exploration_waypoints()

        risk_pct = self.actual_probability * 100.0
        roster_note = (
            f"segment {self.group_index + 1}/{self.group_size}"
            if self.group_size > 1 else "solo coverage"
        )
        rospy.loginfo(
            f"[Drone {drone_id}] EXPLORER assigned to {self.farm_name} ({area_name}) | "
            f"risk {risk_pct:.1f}% | {roster_note} | {len(self.waypoints)} waypoints"
        )
        rospy.loginfo(
            f"[Drone {drone_id}] Risk estimate -> model {self.actual_probability*100:.1f}% | "
            f"onboard {self.measured_probability*100:.1f}% (error {self.risk_error_pct:+.2f}%)"
        )
        self.notes.append(f"Onboard risk error {self.risk_error_pct:+.2f}% against model")
    
    def odom_callback(self, msg):
        """Update current position from odometry"""
        self.current_pose = msg.pose.pose

    def stop_motion(self):
        """Command zero velocity to halt the drone."""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def record_summary(self, status, note=None, final_pos=None):
        """Persist a single mission summary for this drone."""
        if self.result_recorded:
            return

        if note:
            self.notes.append(note)

        if final_pos is None:
            if self.current_pose is not None:
                final_pos = f"({self.current_pose.position.x:.1f}, {self.current_pose.position.y:.1f})"
            else:
                final_pos = f"({self.area_config['x']:.1f}, {self.area_config['y']:.1f})"

        if self.boundary_events and not any('boundary' in entry.lower() for entry in self.notes):
            suffix = 'corrections' if self.boundary_events != 1 else 'correction'
            self.notes.append(f"{self.boundary_events} boundary {suffix}")

        summary = {
            'drone_id': self.drone_id,
            'farm_name': self.farm_name,
            'area_name': self.area_name,
            'actual_probability': self.actual_probability,
            'measured_probability': self.measured_probability,
            'error_pct': self.risk_error_pct,
            'boundary_events': self.boundary_events,
            'final_position': final_pos,
            'status': status,
            'notes': '; '.join(self.notes) if self.notes else ''
        }

        self.result_aggregator.add_result(summary)
        self.result_recorded = True
    
    def generate_exploration_waypoints(self):
        """Generate waypoints to cover the entire area in a grid pattern"""
        center_x = self.area_config['x']
        center_y = self.area_config['y']
        area_size = self.area_config['size']
        altitude = self.exploration_config['flight_altitude']
        spacing = self.exploration_config['waypoint_spacing']
        
        # Calculate area boundaries - STRICT boundaries for colored areas
        half_size = area_size / 2.0
        min_x = center_x - half_size + 0.5  # 0.5m margin to stay within color
        max_x = center_x + half_size - 0.5
        min_y = center_y - half_size + 0.5
        max_y = center_y + half_size - 0.5
        
        # Store boundaries for enforcement during flight
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        rospy.loginfo(f"[Drone {self.drone_id}] Area bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        
        # Generate grid pattern waypoints
        pattern = self.exploration_config['patrol_pattern']
        waypoints = []
        
        if pattern == "grid":
            # Lawn mower pattern
            x = min_x
            y_direction = 1  # 1 for forward, -1 for backward
            
            while x <= max_x + 1e-6:
                if y_direction == 1:
                    # Move from min_y to max_y
                    y = min_y
                    while y <= max_y + 1e-6:
                        waypoints.append((x, y, altitude))
                        y += spacing
                else:
                    # Move from max_y to min_y
                    y = max_y
                    while y >= min_y - 1e-6:
                        waypoints.append((x, y, altitude))
                        y -= spacing
                
                y_direction *= -1  # Reverse direction
                x += spacing
        
        elif pattern == "spiral":
            # Spiral from outside to center
            waypoints = [
                (min_x, min_y, altitude), (max_x, min_y, altitude),
                (max_x, max_y, altitude), (min_x, max_y, altitude),
                (min_x + spacing, min_y + spacing, altitude),
                (max_x - spacing, min_y + spacing, altitude),
                (max_x - spacing, max_y - spacing, altitude),
                (min_x + spacing, max_y - spacing, altitude),
                (center_x, center_y, altitude)
            ]
        
        if not waypoints:
            waypoints = [(center_x, center_y, altitude)]
        else:
            waypoints.append((center_x, center_y, altitude))

        if self.group_size > 1:
            shared_segment = [
                wp for idx, wp in enumerate(waypoints[:-1])
                if idx % self.group_size == self.group_index
            ]
            if not shared_segment and len(waypoints) > 1:
                shared_segment.append(waypoints[self.group_index % (len(waypoints) - 1)])
            shared_segment.append(waypoints[-1])
            self.waypoints = shared_segment
            rospy.loginfo(
                f"[Drone {self.drone_id}] Shared coverage segment {self.group_index + 1}/{self.group_size} "
                f"with {len(self.waypoints)} waypoints"
            )
        else:
            self.waypoints = waypoints
        
        rospy.loginfo(f"[Drone {self.drone_id}] Generated {len(self.waypoints)} waypoints for {self.area_name}")
        rospy.loginfo(
            f"[Drone {self.drone_id}] Coverage area: {self.area_config['color'].upper()} zone at "
            f"({center_x}, {center_y}) - {self.farm_name}"
        )
    
    def get_distance_to_waypoint(self, waypoint):
        """Calculate distance to a waypoint"""
        if self.current_pose is None:
            return float('inf')
        
        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        return sqrt(dx*dx + dy*dy)
    
    def get_angle_to_waypoint(self, waypoint):
        """Calculate angle to a waypoint"""
        if self.current_pose is None:
            return 0
        
        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        return atan2(dy, dx)
    
    def is_within_area_bounds(self):
        """Check if drone is within its assigned area boundaries"""
        if self.current_pose is None:
            return True
        
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        margin = 1.0 # Allow 1.0m drift before triggering safety fail (User Requested Tuning)
        return (self.min_x - margin <= x <= self.max_x + margin) and (self.min_y - margin <= y <= self.max_y + margin)
    
    def explore(self):
        """Execute exploration mission"""
        rate = rospy.Rate(10)  # 10 Hz
        
        center_x = self.area_config['x']
        center_y = self.area_config['y']
        color = self.area_config['color'].upper()
        
        rospy.loginfo(
            f"[Drone {self.drone_id}] [TARGET] Starting {self.farm_name} sweep "
            f"(Area {self.area_name}, {color})..."
        )
        rospy.loginfo(
            f"[Drone {self.drone_id}] Target: {self.farm_name} center at ({center_x}, {center_y})"
        )
        
        while not rospy.is_shutdown() and not self.exploration_complete:
            if self.current_pose is None:
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
                continue
            
            # Check if drone has left its area (safety check)
            if self.current_waypoint_idx > 0 and not self.is_within_area_bounds():
                self.boundary_events += 1
                self.stop_motion()
                rospy.logwarn_throttle(5.0,
                    f"[Drone {self.drone_id}] [WARNING] Outside {self.farm_name} bounds! "
                    f"Returning to center."
                )
                self.notes.append(f"Boundary correction #{self.boundary_events}")
                # CRITICAL: Clear strict path following if we are lost/outside. 
                # Just go to center.
                # Find the center waypoint in our list (usually first or just insert it)
                center_wp = (center_x, center_y, self.exploration_config['flight_altitude'])
                
                # Force immediate return
                dx_global = center_x - self.current_pose.position.x
                dy_global = center_y - self.current_pose.position.y
                
                # Get current yaw
                orientation_q = self.current_pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                
                # Rotate global vector to body frame
                # Body_X (Forward) = dx*cos(yaw) + dy*sin(yaw)
                # Body_Y (Left)    = -dx*sin(yaw) + dy*cos(yaw)
                vx_body = dx_global * cos(yaw) + dy_global * sin(yaw)
                vy_body = -dx_global * sin(yaw) + dy_global * cos(yaw)
                
                cmd = Twist()
                # P-Controller in body frame
                cmd.linear.x = max(-1.0, min(1.0, 1.0 * vx_body))
                cmd.linear.y = max(-1.0, min(1.0, 1.0 * vy_body))
                cmd.linear.z = 0.0 # Maintain altitude
                self.cmd_vel_pub.publish(cmd)
                
                rospy.sleep(0.5) # Allow meaningful movement
                continue
            
            # ------------------------------------------------------------------
            # MPC-Lite / P-Control Movement Logic
            # ------------------------------------------------------------------
            # Check if all waypoints have been visited
            if self.current_waypoint_idx >= len(self.waypoints):
                self.stop_motion()
                self.exploration_complete = True
                rospy.loginfo(f"[CHECK] [Drone {self.drone_id}] Completed {self.farm_name} coverage!")
                break
            
            # Get current target waypoint
            current_waypoint = self.waypoints[self.current_waypoint_idx]
            distance = self.get_distance_to_waypoint(current_waypoint)
            
            if distance < 0.2:
                # Waypoint reached
                self.current_waypoint_idx += 1
                progress = (self.current_waypoint_idx / len(self.waypoints)) * 100
                rospy.loginfo(
                    f"[Drone {self.drone_id}] {self.farm_name}: {self.current_waypoint_idx}/"
                    f"{len(self.waypoints)} waypoints ({progress:.1f}%)"
                )
                continue

            # Calculate velocity command using direct positional error
            cmd = Twist()
            dx = current_waypoint[0] - self.current_pose.position.x
            dy = current_waypoint[1] - self.current_pose.position.y

            # Standard P-Control
            gain = 0.8
            max_linear_vel = 2.0
            
            vx = max(-max_linear_vel, min(max_linear_vel, gain * dx))
            vy = max(-max_linear_vel, min(max_linear_vel, gain * dy))
            
            # DEBUG: Log initial velocity
            # if self.drone_id == 0:  # Only log for one drone to reduce spam
            #    rospy.loginfo(f"Drone 0 POS: {self.current_pose.position.x:.1f},{self.current_pose.position.y:.1f} Bounds: {self.min_x:.1f},{self.max_x:.1f} RAW_VEL: {vx:.2f},{vy:.2f}")

            # Directional Boundary Clamping (The "One-way Mirror" Logic)
            # Directional Boundary Clamping (The "One-way Mirror" Logic)
            # - If inside: allow any movement that keeps us inside (or moves us to edge)
            # - If outside: allow only movement TOWARDS the area. Block movement AWAY.
            
            # X-Axis Bounds
            if self.current_pose.position.x < self.min_x:
                # We are to the LEFT of the area.
                # ONLY allow driving RIGHT (vx > 0). Left (vx < 0) is forbidden.
                if vx < 0:
                    vx = 0.0
            elif self.current_pose.position.x > self.max_x:
                # We are to the RIGHT of the area.
                # ONLY allow driving LEFT (vx < 0). Right (vx > 0) is forbidden.
                if vx > 0:
                    vx = 0.0
            else:
                 # Inside X bounds. Determine soft margin braking.
                 margin = self.boundary_soft_margin
                 # Approaching Left Wall?
                 if (self.current_pose.position.x - self.min_x) < margin and vx < 0:
                     vx = max(vx, -0.3)
                 # Approaching Right Wall?
                 if (self.max_x - self.current_pose.position.x) < margin and vx > 0:
                     vx = min(vx, 0.3)

            # Y-Axis Bounds
            if self.current_pose.position.y < self.min_y:
                # We are BELOW the area.
                # ONLY allow driving UP (vy > 0). Down (vy < 0) is forbidden.
                if vy < 0:
                    vy = 0.0
            elif self.current_pose.position.y > self.max_y:
                # We are ABOVE the area.
                # ONLY allow driving DOWN (vy < 0). Up (vy > 0) is forbidden.
                if vy > 0:
                    vy = 0.0
            else:
                 # Inside Y bounds. Soft margin braking.
                 margin = self.boundary_soft_margin
                 # Approaching Bottom Wall?
                 if (self.current_pose.position.y - self.min_y) < margin and vy < 0:
                     vy = max(vy, -0.3)
                 # Approaching Top Wall?
                 if (self.max_y - self.current_pose.position.y) < margin and vy > 0:
                     vy = min(vy, 0.3)

            # Z-axis control (Altitude P-Controller)
            target_z = self.exploration_config.get('flight_altitude', 2.0)
            # Safety clamp for target Z
            target_z = max(0.5, min(15.0, target_z))
            
            z_err = target_z - self.current_pose.position.z
            vz = max(-1.0, min(1.0, 1.5 * z_err))
            
            cmd.linear.x = vx
            cmd.linear.y = vy
            cmd.linear.z = vz
            cmd.angular.z = 0.0

            self.cmd_vel_pub.publish(cmd)

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

        if self.exploration_complete:
            self.record_summary('complete', note='Returned to center')
        elif rospy.is_shutdown():
            self.record_summary('aborted', note='ROS shutdown during mission')
        else:
            self.record_summary('stopped', note='Exploration halted early')

        self.stop_motion()
        rospy.loginfo(
            f"[Drone {self.drone_id}] Risk report -> actual {self.actual_probability*100:.1f}% | "
            f"onboard {self.measured_probability*100:.1f}% | error {self.risk_error_pct:+.2f}% | "
            f"boundary corrections {self.boundary_events}"
        )


class BackupDrone:
    """Backup drone that stays at starting position"""
    def __init__(self, drone_id, start_pos, result_aggregator=None, measurement_noise=0.05):
        self.drone_id = drone_id
        self.start_pos = start_pos
        self.current_pose = None
        self.result_aggregator = result_aggregator
        self.measurement_noise = abs(measurement_noise)

        self.actual_probability = 0.0
        self.measured_probability = clamp(
            self.actual_probability * (1.0 + random.uniform(-self.measurement_noise, self.measurement_noise))
        )
        self.risk_error_pct = (self.measured_probability - self.actual_probability) * 100.0
        # Subscribe to odometry
        # Use absolute path to ensure we hit the global topic
        odom_topic = f"/drone_{drone_id}/odom"
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback
        )
        
        # Publisher for velocity control
        cmd_vel_topic = f"/drone_{drone_id}/cmd_vel"
        self.cmd_vel_pub = rospy.Publisher(
            cmd_vel_topic, Twist, queue_size=1
        )
        
        rospy.loginfo(f"[Drone {drone_id}] BACKUP - Holding position at start")
        rospy.loginfo(
            f"[Drone {drone_id}] Risk estimate -> idle staging asset, onboard {self.measured_probability*100:.1f}%"
        )
    
    def odom_callback(self, msg):
        """Update current position from odometry"""
        self.current_pose = msg.pose.pose
    
    def hold_position(self):
        """Maintain position at starting location"""
        rate = rospy.Rate(5)  # 5 Hz (less frequent than explorers)
        
        while not rospy.is_shutdown():
            # Check if we have a valid pose
            if self.current_pose is None:
                if self.drone_id == 0 and random.random() < 0.05:
                    rospy.logwarn(f"[Drone {self.drone_id}] Waiting for odometry on 'drone_{self.drone_id}/odom'...")
                rospy.sleep(0.1)
                continue
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            
            # Calculate distance from start position
            dx = self.start_pos['x'] - self.current_pose.position.x
            dy = self.start_pos['y'] - self.current_pose.position.y
            distance = sqrt(dx*dx + dy*dy)
            
            # If drifted too far from start, return to position
            if distance > 1.0:  # More than 1m from start
                cmd = Twist()
                cmd.linear.x = min(0.5, distance * 0.3)
                angle = atan2(dy, dx)
                cmd.angular.z = angle * 0.5
                self.cmd_vel_pub.publish(cmd)
            else:
                # Stay still
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)

            if self.result_aggregator and self.current_pose is not None and not rospy.is_shutdown():
                summary = {
                    'drone_id': self.drone_id,
                    'farm_name': 'Staging Pad',
                    'area_name': 'staging',
                    'actual_probability': self.actual_probability,
                    'measured_probability': self.measured_probability,
                    'error_pct': self.risk_error_pct,
                    'boundary_events': 0,
                    'final_position': f"({self.current_pose.position.x:.1f}, {self.current_pose.position.y:.1f})" if self.current_pose else '---',
                    'status': 'reserve',
                    'notes': 'Idle drought reserve drone'
                }
                self.result_aggregator.add_result(summary)
                self.result_aggregator = None
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main():
    rospy.init_node('area_explorer_node')
    
    # Load configuration
    config_path = rospy.get_param('~config_path', 
                                   os.path.join(os.path.dirname(__file__), '../config/areas.yaml'))
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        rospy.logerr(f"Failed to load config: {e}")
        return
    
    num_drones = config['num_drones']
    areas = config['areas']
    start_pos = config['start_position']
    exploration_config = config['exploration']
    allocation_cfg = config.get('allocation', {})
    measurement_noise = allocation_cfg.get('measurement_noise', 0.12)
    boundary_soft_margin = allocation_cfg.get('boundary_soft_margin', 0.4)
    idle_measurement_noise = allocation_cfg.get('idle_measurement_noise', 0.05)

    aggregator = MissionAggregator()    # 2. Analyze risk and allocate
    rospy.loginfo("[Main] Analyzing drought risk for all areas...")
    area_profiles = {name: analyze_drought_risk(name, cfg) for name, cfg in areas.items()}
    
    rospy.loginfo("[Main] Allocating drones to areas (Advanced Logic)...")
    
    # Convert config to Area objects for the allocator
    area_objects = []
    for name, cfg in areas.items():
        profile = area_profiles[name]
        area_objects.append(Area(
            area_id=name,
            drought_probability=profile['probability'],
            size_m2=cfg.get('size', 10.0)**2 * 3.14, # Approx area
            coverage_radius=cfg.get('size', 10.0)/2,
            x_center=cfg.get('x', 0),
            y_center=cfg.get('y', 0),
            crop_type=cfg.get('crop', 'unknown')
        ))
        
    # Create Drone objects
    drone_objects = [Drone(drone_id=i) for i in range(num_drones)]
    
    # Initialize Allocator
    allocator = DynamicDroneAllocator(
        total_drones=num_drones,
        total_areas=len(areas),
        min_drones_per_area=allocation_cfg.get('min_drones_per_area', 1),
        max_drones_per_area=allocation_cfg.get('max_drones_per_area', 3),
        reserve_percentage=0.1 # Default 10% reserve for auditors
    )
    
    # Execute Allocation
    result = allocator.allocate_drones(area_objects, drone_objects)
    
    allocation_counts = {aid: len(dids) for aid, dids in result.allocations.items()}
    reserve_drones = len(result.reserve_drones)
    
    # Print Summary Table
    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("ALLOCATION SUMMARY (PHASE 2-4 LOGIC)")
    rospy.loginfo("="*60)
    summary = result.allocation_summary
    for entry in summary['allocation_table']:
        risk_label = "HIGH" if entry['probability'] > 0.7 else "MED" if entry['probability'] > 0.4 else "LOW"
        rospy.loginfo(f"Area {entry['area_id']:<6} | Risk {entry['probability']:.2f} ({risk_label}) | Drones: {entry['drones_assigned']} {entry['drone_ids']}")
    rospy.loginfo(f"Total Allocated: {summary['total_allocated']} | Reserves: {summary['total_reserves']}")
    rospy.loginfo("="*60 + "\n")

    rospy.loginfo("[Main] Initializing drone explorers...")
    
    # Map back to explorer_plan format
    full_plan = [None] * num_drones
    
    # 1. Explorers
    for area_id, drone_ids in result.allocations.items():
        profile = area_profiles[area_id]
        for idx, did in enumerate(drone_ids):
            full_plan[did] = {
                'role': 'explorer',
                'area': area_id,
                'group_index': idx,
                'group_size': len(drone_ids),
                'probability': profile['probability']
            }
            
    # 2. Reserves
    for did in result.reserve_drones:
        full_plan[did] = {'role': 'backup'}
        
    # Safety fallback
    for i in range(num_drones):
        if full_plan[i] is None:
            full_plan[i] = {'role': 'backup'}
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("         MULTI-DRONE FARMLAND EXPLORATION")
    rospy.loginfo("=" * 60)

    # 1. Load configuration
    rospy.loginfo("[Main] Loading area configuration...")
    config_path = rospy.get_param('~area_config_path', "/home/ros/catkin_ws/src/multi_drone_sim/config/areas.yaml")
    rospy.loginfo("")
    rospy.loginfo("FARMLAND DROUGHT PRIORITISATION:")
    rospy.loginfo("-" * 60)

    ordered_by_risk = sorted(areas.keys(), key=lambda name: area_profiles[name]['probability'], reverse=True)
    for area_name in ordered_by_risk:
        profile = area_profiles[area_name]
        area_cfg = areas[area_name]
        farm_name = area_cfg.get('name', area_name)
        assigned = allocation_counts.get(area_name, 0)
        rospy.loginfo(
            f"  {farm_name:<18} (Area {area_name}, {area_cfg.get('color', 'n/a'):>6}) | "
            f"P(drought)={profile['probability']*100:.1f}% | Drones={assigned}"
        )
    rospy.loginfo(f"  Backup drones: {backup_count}")
    
    # Wait for simulation to stabilize
    rospy.loginfo("Waiting for simulation to start...")
    rospy.sleep(5.0)
    
    # Create drone controllers
    explorers = []
    backups = []
    
    for drone_id in range(num_drones):
        plan = full_plan[drone_id]
        if plan['role'] == 'explorer':
            area_name = plan['area']
            area_config = areas[area_name]
            farm_name = area_config.get('name', area_name)
            risk_pct = plan['probability'] * 100.0
            rospy.loginfo(
                f"  Drone {drone_id} -> {farm_name} (Area {area_name}) | "
                f"segment {plan['group_index'] + 1}/{plan['group_size']} | risk {risk_pct:.1f}%"
            )
            explorer = DroneExplorer(
                drone_id,
                area_name,
                area_config,
                exploration_config,
                start_pos,
                aggregator,
                measurement_noise=measurement_noise,
                boundary_soft_margin=boundary_soft_margin,
                group_index=plan['group_index'],
                group_size=plan['group_size'],
                drought_probability=plan.get('probability')
            )
            explorers.append(explorer)
        else:
            rospy.loginfo(f"  Drone {drone_id} -> BACKUP (stays at start)")
            backup = BackupDrone(
                drone_id,
                start_pos,
                result_aggregator=aggregator,
                measurement_noise=idle_measurement_noise
            )
            backups.append(backup)
    
    rospy.loginfo("-" * 60)
    
    rospy.loginfo(f"Active Explorers: {len(explorers)}")
    rospy.loginfo(f"Backup Drones: {len(backups)}")
    rospy.loginfo("-" * 60)

    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'drought_allocation.log'))
    mission_results = aggregator.get_results()
    build_allocation_report(
        report_path,
        areas,
        area_profiles,
        allocation_counts,
        full_plan,
        mission_results=mission_results
    )
    rospy.loginfo(f"Allocation report written to {report_path}")
    
    # Wait for odometry
    rospy.loginfo("Waiting for odometry data...")
    rospy.sleep(3.0)
    
    # Start exploration threads
    rospy.loginfo("Starting area exploration missions...")
    rospy.loginfo("=" * 60)
    
    threads = []
    
    # Start explorer threads
    for explorer in explorers:
        thread = threading.Thread(target=explorer.explore)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Start backup threads
    for backup in backups:
        thread = threading.Thread(target=backup.hold_position)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all explorers to complete
    for explorer in explorers:
        while not explorer.exploration_complete and not rospy.is_shutdown():
            rospy.sleep(1.0)
    
    completed = sum(1 for explorer in explorers if explorer.exploration_complete)
    interrupted = len(explorers) - completed

    rospy.loginfo("=" * 60)
    if interrupted == 0:
        rospy.loginfo("[CHECK] ALL EXPLORATION MISSIONS COMPLETED!")
        rospy.loginfo(f"  - {completed} drones have fully explored their assigned areas")
    else:
        rospy.logwarn("! EXPLORATION MISSIONS ENDED EARLY")
        rospy.loginfo(f"  - {completed} drones completed coverage")
        rospy.loginfo(f"  - {interrupted} drones returned early or were interrupted")
    rospy.loginfo(f"  - {len(backups)} backup drones remain at starting position")
    rospy.loginfo("=" * 60)

    for thread in threads:
        thread.join(timeout=1.0)

    rospy.sleep(0.5)

    mission_results = aggregator.get_results()
    build_allocation_report(
        report_path,
        areas,
        area_profiles,
        allocation_counts,
        full_plan,
        mission_results=mission_results
    )
    rospy.loginfo("Allocation report updated with mission observations")
    
    # Keep node running
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
