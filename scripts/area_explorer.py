#!/usr/bin/env python3

import rospy
import yaml
import os
import threading
import random
from datetime import datetime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import math
from math import sqrt, atan2, exp, pi, cos, sin
import math
from math import sqrt, atan2, exp, pi, cos, sin
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker

# New Hybrid Algo
from algo3_sim import GridWaypointManager, Algo3Hybrid
import sys
# Ensure we can import local modules - handle both direct execution and catkin wrapper
script_dir = os.path.dirname(os.path.abspath(__file__))
if 'devel/lib' in script_dir:
    # Running through catkin wrapper, find the actual scripts directory
    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), 'src', 'multi_drone_sim', 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
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


def write_mission_summary(summary_path, areas_cfg, full_plan, mission_results):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append('MULTI-DRONE MISSION SUMMARY')
    lines.append(f'Generated (UTC): {timestamp}')
    lines.append('')
    lines.append('Central Agent: Tower beacon stopped after mission_complete (HELLO/ACK queue drained)')
    lines.append('UGV Chargers: Parked after mission_complete signal')
    lines.append('')
    lines.append('Area centers and assignments:')
    lines.append('Area    Center (x,y)   Color     Drones')
    lines.append('-----   -------------  --------  -------------------')
    for area_name, cfg in areas_cfg.items():
        assigned = [i for i, plan in enumerate(full_plan) if plan['role'] == 'explorer' and plan['area'] == area_name]
        lines.append(
            f"{area_name:<7} ({cfg.get('x',0):>5.1f},{cfg.get('y',0):>5.1f})  {cfg.get('color','n/a'):<8}  {assigned}"
        )

    lines.append('')
    lines.append('Drone docking/finish states:')
    lines.append('Drone  Role       Area       Final (x,y)     Status      Notes')
    lines.append('-----  ---------  ---------  --------------  ----------  ---------------------------')

    # Index mission results by drone id for quick lookup
    result_map = {entry['drone_id']: entry for entry in mission_results}

    for drone_id, plan in enumerate(full_plan):
        role = plan.get('role', 'n/a')
        area = plan.get('area', 'n/a')
        if drone_id in result_map:
            entry = result_map[drone_id]
            final_pos = entry.get('final_position', '--')
            status = entry.get('status', 'n/a')
            notes = entry.get('notes', '')
        else:
            final_pos = '--'
            status = 'n/a'
            notes = ''

        lines.append(
            f"{drone_id:<5}  {role:<9}  {area:<9}  {final_pos:<14}  {status:<10}  {notes}"
        )

    lines.append('')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

from std_msgs.msg import Float32, Bool

class Battery:
    """
    Simulates LiPo battery discharge.
    Model based on: Karapetyan et al., ICRA 2024 (simplified)
    """
    def __init__(self, capacity_mah=800, start_voltage=12.6):
        self.capacity_mah = capacity_mah
        self.current_charge = capacity_mah
        self.voltage = start_voltage
        
        # Power consumption params (Amps)
        self.idle_current = 0.5  # Avionics only
        self.hover_current = 15.0 # Motors hovering
        self.flight_current_slope = 5.0 # Amps per m/s velocity
        
    def consume(self, velocity, dt):
        """
        Update charge based on flight state.
        velocity: current speed in m/s
        dt: time step in seconds
        """
        if self.current_charge <= 0:
            return 0.0
            
        # Current draw model: I = I_hover + k * v
        draw = self.hover_current + (self.flight_current_slope * velocity)
        if velocity < 0.1: # Near stationary/hover
            draw = self.hover_current
            
        # discharge (Ah) = draw (A) * dt (h)
        drain_mah = draw * (dt / 3600.0) * 1000.0
        self.current_charge -= drain_mah
        
        return self.get_percentage()
        
    def get_percentage(self):
        return max(0.0, (self.current_charge / self.capacity_mah) * 100.0)

    def recharge(self, percentage_target):
        """Recharge to a specific percentage (e.g., 100.0)"""
        self.current_charge = (percentage_target / 100.0) * self.capacity_mah



class AreaCoverageController:
    """
    Manages the Algo 3 Simulation for a single farmland area.
    Acts as a bridge between ROS (DroneExplorer) and the Algo Library.
    """
    def __init__(self, area_name, area_config, drone_explorers):
        self.area_name = area_name
        self.area_config = area_config
        # Sort Drones by X-coordinate (Left -> Right)
        # This ensures Drone 0 gets the Left Partition and Drone 1 gets the Right Partition,
        # minimizing cross-over flight paths.
        drone_explorers.sort(key=lambda d: d.current_pose.position.x if d.current_pose else d.start_pos[0])
        self.drone_explorers = drone_explorers

        
        # Setup Grid
        # Area size is length of side (square)
        self.side_length = area_config['size']
        self.center_x = area_config['x']
        self.center_y = area_config['y']
        
        # Calculate origin (bottom-left) for coordinate transform
        self.origin_x = self.center_x - (self.side_length / 2.0)
        self.origin_y = self.center_y - (self.side_length / 2.0)
        
        # Initialize Grid Manager & Hybrid Logic
        self.grid_manager = GridWaypointManager(
            side_length=self.side_length,
            coverage_radius=1.5,
            origin_x=self.origin_x,
            origin_y=self.origin_y
        )
        self.algo = Algo3Hybrid(self.grid_manager, num_drones=len(self.drone_explorers))
        
        # Map Local Index -> Global ID
        self.local_to_global = {}
        for i, explorer in enumerate(self.drone_explorers):
            # explorer is NotificationDrone or similar
            self.local_to_global[i] = getattr(explorer, 'drone_id', f"Unknown-{i}")

        self.last_step_time = rospy.Time.now()
        
        # NOTE: Drones fly directly to the nearest unvisited waypoint in their partition
        # No explicit "APPROACH" state is needed as GOMWC handles distance naturally.


    def step(self):
        """
        Execute one step of the algorithm logic.
        Drones sync position, Algo determines target, Drones execute.
        """
        # 1. Update Positions & Algo State
        for i, explorer in enumerate(self.drone_explorers):
            if explorer.current_pose:
                local_x = explorer.current_pose.position.x - self.origin_x
                local_y = explorer.current_pose.position.y - self.origin_y
                
                # Update Algo with position (Checks for visited waypoints internally)
                self.algo.update_drone_pose(i, local_x, local_y)

        # 2. Decision Making
        msg = self.algo.step()
        
        # 3. Execute Commands
        finished_drones = 0
        
        for i, explorer in enumerate(self.drone_explorers):
            target = self.algo.get_target_coords(i)
            
            if target:
                tx, ty = target
                # Transform to global
                global_tx = tx + self.origin_x
                global_ty = ty + self.origin_y
                explorer.update_control(global_tx, global_ty)
            else:
                 # No target assigned (Finished or Idle)
                 finished_drones += 1
                 if explorer.current_pose:
                     explorer.update_control(explorer.current_pose.position.x, explorer.current_pose.position.y)

        # 4. Stats Logging
        v, t, pct = self.grid_manager.get_progress_stats()
        drone_stats = self.grid_manager.get_drone_stats()
        
        # Format drone stats string: "Drone 5: 12, Drone 6: 15"
        # drone_stats keys are Local IDs (0, 1...)
        parts = []
        for local_id, cnt in drone_stats.items():
            global_id = self.local_to_global.get(local_id, f"?{local_id}")
            parts.append(f"Drone {global_id}: {cnt}pts")
            
        d_stats_str = ", ".join(parts)
        
        rospy.loginfo_throttle(5, f"[{self.area_name}] Area Coverage: {pct:.1f}% ({v}/{t}) | {d_stats_str}")

    def is_complete(self):
        v, t, pct = self.grid_manager.get_progress_stats()
        return pct >= 99.9

    def get_progress(self):
        v, t, pct = self.grid_manager.get_progress_stats()
        return pct


class DroneExplorer:
    """Explorer drone that patrols a specific area (ROS Interface)"""

    def __init__(self, drone_id, area_name, area_config, exploration_config, start_pos,
                 result_aggregator, measurement_noise=0.1, boundary_soft_margin=0.3,
                 group_index=0, group_size=1, drought_probability=None):
        self.drone_id = drone_id
        self.area_name = area_name
        self.area_config = area_config
        self.exploration_config = exploration_config
        self.start_pos = start_pos
        self.current_pose = None
        
        # Algo 3 - No static waypoints
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
        
        # Battery Simulation
        self.battery = Battery(capacity_mah=1500)
        self.last_battery_time = rospy.Time.now()
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher(f'/drone_{drone_id}/cmd_vel', Twist, queue_size=10)
        self.battery_pub = rospy.Publisher(f'/drone_{drone_id}/battery', Float32, queue_size=10)
        self.odom_sub = rospy.Subscriber(f'/drone_{drone_id}/odom', Odometry, self.odom_callback)
        self.charge_sub = rospy.Subscriber(f'/drone_{drone_id}/charge_cmd', Float32, self.charge_callback)
        self.marker_pub = rospy.Publisher(f'/drone_{drone_id}/cone_marker', Marker, queue_size=1)
        
        # Assign altitude (1.5m - 2.0m as requested)
        self.flight_altitude = 2.0

        risk_pct = self.actual_probability * 100.0
        rospy.loginfo(
            f"[Drone {drone_id}] EXPLORER assigned to {self.farm_name} ({area_name}) | "
            f"Risk {risk_pct:.1f}% | Algo 3 Controlled"
        )

    def charge_callback(self, msg):
        rospy.loginfo(f"[Drone {self.drone_id}] Receiving Charge: {msg.data}%")
        self.battery.recharge(msg.data)

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def stop_motion(self):
        # Land and disarm (simulated by zero vel)
        # Send simple land command (negative z)
        cmd = Twist()
        cmd.linear.z = -0.5 # Descend
        for _ in range(10): # Send multiple times to ensure receipt
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)

    def record_summary(self, status, note=None, final_pos=None):
        if self.result_recorded: return
        if note: self.notes.append(note)
        if final_pos is None:
            if self.current_pose:
                final_pos = f"({self.current_pose.position.x:.1f}, {self.current_pose.position.y:.1f})"
            else:
                final_pos = f"({self.area_config['x']:.1f}, {self.area_config['y']:.1f})"

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

    def update_control(self, target_x, target_y):
        """
        Move towards target waypoint using P-Controller.
        Called by AreaCoverageController.
        """
        if self.current_pose is None: 
            rospy.logwarn_throttle(5, f"[Drone {self.drone_id}] No Odometry!")
            return
        
        # DEBUG: Confirm we are receiving commands
        rospy.loginfo_throttle(5, f"[Drone {self.drone_id}] Moving to ({target_x:.1f}, {target_y:.1f})")

        # Battery Logic
        current_time = rospy.Time.now()
        dt = (current_time - self.last_battery_time).to_sec()
        self.last_battery_time = current_time
        
        # Estimate speed
        # For simulation, just assume moving if not at target
        speed = 2.0 
        bat_pct = self.battery.consume(speed, dt)
        self.battery_pub.publish(bat_pct)
        
        if bat_pct < 20.0:
             rospy.logwarn_throttle(10, f"Drone {self.drone_id} Low Battery: {bat_pct:.1f}%")
             # Ideally return to base logic here, but for now just warn

        # P-Control Logic
        dx = target_x - self.current_pose.position.x
        dy = target_y - self.current_pose.position.y
        dist = sqrt(dx*dx + dy*dy)
        
        # Altitude Control
        dz = self.flight_altitude - self.current_pose.position.z
        
        cmd = Twist()
        # Gain
        kp_xy = 1.0
        kp_z = 1.0
        
        # Global velocities
        vx = max(-1.0, min(1.0, kp_xy * dx))
        vy = max(-1.0, min(1.0, kp_xy * dy))
        vz = max(-0.5, min(0.5, kp_z * dz))
        
        # DEBUG: Log command
        # if self.drone_id == 0:
        #    rospy.loginfo_throttle(1, f"CMD D0: Tgt({target_x:.1f},{target_y:.1f}) Pos({self.current_pose.position.x:.1f},{self.current_pose.position.y:.1f}) Vz={vz:.2f}")
        
        # Transform global velocity to body frame for drone
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf_trans.euler_from_quaternion(orientation_list)
        
        # Rotate vector by -yaw
        vx_body = vx * cos(yaw) + vy * sin(yaw)
        vy_body = -vx * sin(yaw) + vy * cos(yaw)

        cmd.linear.x = vx_body
        cmd.linear.y = vy_body
        # Boost Z gain and limit for faster takeoff
        z_boost = 2.0 * dz
        cmd.linear.z = max(-1.0, min(1.0, z_boost))
        
        # Heading Control: Face target
        desired_yaw = atan2(dy, dx)
        yaw_err = desired_yaw - yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        if dist > 0.5: # only turn if significantly far
            cmd.angular.z = max(-1.0, min(1.0, 1.0 * yaw_err))
        
        # DEBUG: Print Z command if not taking off
        # if self.drone_id == 0:
        #     rospy.loginfo_throttle(1, f"Alt Err: {z_err:.2f} CmdZ: {cmd.linear.z:.2f}")

        self.cmd_vel_pub.publish(cmd)
        
        # Publish Visual Cone
        self.publish_cone()

    def publish_cone(self):
        """Publish a semi-transparent cone marker representing the sensor FoV"""
        if not self.current_pose: return

        marker = Marker()
        # Using "world" frame is safer if drone frames aren't reliable
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "fov_cone"
        marker.id = self.drone_id
        marker.type = Marker.CYLINDER # Approximates a beam/cone
        marker.action = Marker.ADD
        
        # Position: Cylinder from drone to ground
        h = self.current_pose.position.z
        radius = max(0.2, h * 0.5) # Tan(theta) approx
        
        marker.pose.position.x = self.current_pose.position.x
        marker.pose.position.y = self.current_pose.position.y
        marker.pose.position.z = h / 2.0 # Center of cylinder
        
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = h
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.5 # Semi-transparent Cyan
        
        self.marker_pub.publish(marker)



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
    
    def land(self):
        """Land the drone"""
        rospy.loginfo(f"[Drone {self.drone_id}] Landing...")
        rate = rospy.Rate(10)
        
        # Simple open-loop descent
        for _ in range(50):
            cmd = Twist()
            cmd.linear.z = -0.5
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        # Stop
        self.cmd_vel_pub.publish(Twist())

class CoverageGrid:
    def __init__(self, min_x, max_x, min_y, max_y, resolution=0.5):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.res = resolution
        
        self.cols = int((max_x - min_x) / resolution) + 1
        self.rows = int((max_y - min_y) / resolution) + 1
        
        # 0 = Uncovered, 1 = Covered
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.total_cells = self.rows * self.cols
        self.covered_cells = 0

    def update(self, x, y, radius):
        """Mark cells within radius of (x,y) as covered"""
        # Convert world circle to grid indices
        # Simple bounding box check for speed
        
        # Grid coordinates of the center
        gx = int((x - self.min_x) / self.res)
        gy = int((y - self.min_y) / self.res)
        
        # Grid radius
        gr = int(radius / self.res)
        
        # Bounding box of the circle on the grid
        y_min = max(0, gy - gr)
        y_max = min(self.rows, gy + gr + 1)
        x_min = max(0, gx - gr)
        x_max = min(self.cols, gx + gr + 1)
        
        # Iterate over bounding box
        for r in range(y_min, y_max):
            for c in range(x_min, x_max):
                if self.grid[r, c] == 1:
                    continue
                
                # Check distance
                # cell center in world coords
                cell_wx = self.min_x + c * self.res + self.res/2
                cell_wy = self.min_y + r * self.res + self.res/2
                
                if (cell_wx - x)**2 + (cell_wy - y)**2 <= radius**2:
                    self.grid[r, c] = 1
                    self.covered_cells += 1
                    
    def get_progress(self):
        if self.total_cells == 0: return 0.0
        return float(self.covered_cells) / self.total_cells

    def explore(self):
        """
        Main exploration loop
        Executes the coverage path while monitoring battery and risk
        """
        rate = rospy.Rate(10)
        self.mission_active = True
        waypoint_idx = 0
        
        # Initialize Dynamic Coverage Grid
        # Use simple heuristic: Cone radius = Altitude * 0.625 (matches visual mesh scale)
        self.coverage_grid = CoverageGrid(self.min_x, self.max_x, self.min_y, self.max_y, resolution=0.5)
        self.last_progress_log = 0.0 # Initialize for progress logging
        
        rospy.loginfo(f"[Drone {self.drone_id}] Starting dynamic coverage sweep...")
        
        while not rospy.is_shutdown() and self.mission_active:
            # 1. Check Battery
            if self.battery_level < 20.0:
                rospy.logwarn(f"[Drone {self.drone_id}] Low battery ({self.battery_level:.1f})! Returning to base.")
                self.return_to_base()
                break
                
            # 2. Get current pose
            if self.current_pose is None:
                rate.sleep()
                continue
                
            curr_x = self.current_pose.position.x
            curr_y = self.current_pose.position.y
            curr_z = self.current_pose.position.z
            
            # --- DYNAMIC COVERAGE LOGIC ---
            # Calculate cone radius based on altitude
            # Visual cone scale in SDF is 2.5/4.0 ~= 0.625
            fov_radius = max(0.5, curr_z * 0.625)
            
            # Update grid
            self.coverage_grid.update(curr_x, curr_y, fov_radius)
            progress = self.coverage_grid.get_progress()
            
            # Log progress periodically (every 10%)
            if progress >= self.last_progress_log + 0.10:
                 rospy.loginfo(f"[Drone {self.drone_id}] Dynamic Coverage: {progress*100:.1f}% (Alt: {curr_z:.1f}m, Radius: {fov_radius:.1f}m)")
                 self.last_progress_log = progress

            # 3. Navigation Logic (Waypoint Following)
            if waypoint_idx < len(self.waypoints):
                target_x, target_y, target_z = self.waypoints[waypoint_idx]
                
                # Simple P-Controller
                dist, angle = self.get_dist_angle(target_x, target_y)
                
                # Calculate errors
                yaw = self.current_yaw
                angle_diff = angle - yaw
                
                # Customize for "Dynamic Vision": Ensure we maintain altitude for coverage
                # If we are too low, coverage is small. If too high, resolution drops (not modeled here, but good conceptually)
                
                # Check arrival
                if dist < 0.5:
                    waypoint_idx += 1
                    # rospy.loginfo(f"[Drone {self.drone_id}] Reached waypoint {waypoint_idx}/{len(self.waypoints)}")
                
                # Control output
                cmd = Twist()
                
                # Rotate to face target
                # Normalize angle
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                
                if abs(angle_diff) > 0.5:
                    cmd.angular.z = max(-0.5, min(0.5, angle_diff))
                else:
                    cmd.linear.x = min(1.0, dist)  # Move forward
                    cmd.angular.z = angle_diff     # Fine tune heading
                    
                    # Altitude control
                    z_err = target_z - curr_z
                    cmd.linear.z = max(-0.5, min(0.5, z_err))
                
                self.cmd_vel_pub.publish(cmd)
                
            else:
                # Mission Complete
                rospy.loginfo(f"[CHECK] [Drone {self.drone_id}] Completed {self.farm_name} coverage! Final Dynamic Coverage: {progress*100:.1f}%")
                
                # Record results
                self.record_mission_result(final_pos=f"({curr_x:.1f}, {curr_y:.1f})", status="success")
                
                self.mission_active = False
                self.notes = []
                self.last_progress_log = 0.0
                self.return_to_base()
            
            rate.sleep()
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

    mission_pub = rospy.Publisher('/mission_complete', Bool, queue_size=1, latch=True)
    
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
    
    rospy.loginfo("[Main] Allocating drones: exactly 2 explorers per farmland + 1 standby...")

    # Deterministic 2-per-area allocation with a single standby at spawn
    standby_desired = 1
    area_names = list(areas.keys())
    total_required = len(area_names) * 2
    if num_drones < total_required + standby_desired:
        rospy.logwarn(
            "Fleet size is below 2-per-area + standby. Reducing standby to fit available drones."
        )
        standby_desired = max(0, num_drones - total_required)

    available_drones = list(range(num_drones))
    full_plan = [None] * num_drones
    allocation_counts = {}

    for area_name in area_names:
        group_assignments = []

        # Keep standby_desired drones unassigned until end
        while len(group_assignments) < 2 and len(available_drones) > standby_desired:
            group_assignments.append(available_drones.pop(0))

        group_size = len(group_assignments)
        allocation_counts[area_name] = group_size

        if group_size < 2:
            rospy.logwarn(
                f"[Main] Only {group_size} drone(s) available for {area_name}; coverage will be partial."
            )

        for idx, drone_id in enumerate(group_assignments):
            full_plan[drone_id] = {
                'role': 'explorer',
                'area': area_name,
                'group_index': idx,
                'group_size': group_size,
                'probability': area_profiles[area_name]['probability'],
                'role_label': 'explorer'
            }

    # Remaining drones become standby/backups at the spawn pad
    standby_drones = []
    while available_drones:
        drone_id = available_drones.pop(0)
        if len(standby_drones) < standby_desired:
            standby_drones.append(drone_id)
        full_plan[drone_id] = {
            'role': 'backup',
            'area': 'staging',
            'role_label': 'backup'
        }

    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("ALLOCATION SUMMARY (FIXED 2-PER-AREA)")
    rospy.loginfo("="*60)
    for area_name in area_names:
        assigned = [i for i, plan in enumerate(full_plan) if plan['role'] == 'explorer' and plan['area'] == area_name]
        risk = area_profiles[area_name]['probability']
        risk_label = "HIGH" if risk > 0.7 else "MED" if risk > 0.4 else "LOW"
        rospy.loginfo(
            f"Area {area_name:<6} | Risk {risk:.2f} ({risk_label}) | Drones: {len(assigned)} {assigned}"
        )
    rospy.loginfo(f"Total Explorers: {sum(1 for p in full_plan if p['role']=='explorer')}")
    rospy.loginfo(f"Standby/Backups: {sum(1 for p in full_plan if p['role']=='backup')}")
    rospy.loginfo("="*60 + "\n")

    rospy.loginfo("[Main] Initializing drone explorers...")
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("         MULTI-DRONE FARMLAND EXPLORATION")
    rospy.loginfo("=" * 60)

    # 1. Load configuration
    rospy.loginfo("[Main] Loading area configuration...")
    config_path = rospy.get_param('~area_config_path', "/home/ros/catkin_ws/src/multi_drone_sim/config/areas.yaml")
    rospy.loginfo("")
    rospy.loginfo("FARMLAND DROUGHT PRIORITISATION:")
    rospy.loginfo("-" * 60)

    reserve_drones = sum(1 for p in full_plan if p['role'] == 'backup')
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
    rospy.loginfo(f"  Backup drones: {reserve_drones}")
    
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
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'mission_summary.log'))
    mission_results = aggregator.get_results()
    build_allocation_report(
        report_path,
        areas,
        area_profiles,
        allocation_counts,
        full_plan,
        mission_results=mission_results
    )
    write_mission_summary(
        summary_path,
        areas,
        full_plan,
        mission_results
    )
    rospy.loginfo(f"Allocation report written to {report_path}")
    rospy.loginfo(f"Mission summary written to {summary_path}")
    

    # Wait for odometry
    rospy.loginfo("Waiting for odometry data...")
    rospy.sleep(3.0)
    
    # Initialize Area Managers (Algo 3)
    area_controllers = []
    
    # Group explorers by area
    area_groups = {}
    for explorer in explorers:
        if explorer.area_name not in area_groups:
            area_groups[explorer.area_name] = []
        area_groups[explorer.area_name].append(explorer)
    
    for area_name, explorer_list in area_groups.items():
        area_cfg = areas[area_name]
        controller = AreaCoverageController(area_name, area_cfg, explorer_list)
        area_controllers.append(controller)

    # Start exploration missions (Centralized Loop)
    rospy.loginfo("Starting Algo 3 coverage simulation...")
    rospy.loginfo("=" * 60)
    
    # Start backup threads (they are independent)
    threads = []
    for backup in backups:
        thread = threading.Thread(target=backup.hold_position)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    rate = rospy.Rate(10) # 10Hz
    start_time = rospy.Time.now()
    
    while not rospy.is_shutdown():
        # Step each area controller
        active_areas = 0
        
        for controller in area_controllers:
            if not controller.is_complete():
                controller.step()
                active_areas += 1
            else:
                # Optional: Command drones to land or hold?
                # For now, they will stop receiving updates and hover (cmd_vel timeout or similar)
                # But best to command hover explicitly
                for drone in controller.drone_explorers:
                    if not drone.exploration_complete:
                         drone.exploration_complete = True
                         drone.stop_motion()
                         drone.record_summary('complete', "Algo 3 Coverage Complete") 
                         rospy.loginfo(f"Drone {drone.drone_id} finished. Landing...")

        # Calculate Total System Coverage
        total_pts = 0
        total_covered = 0
        for controller in area_controllers:
            v, t, _ = controller.grid_manager.get_progress_stats()
            total_covered += v
            total_pts += t
            
        total_pct = (total_covered / total_pts * 100.0) if total_pts > 0 else 0.0
        rospy.loginfo_throttle(10, f"[SYSTEM] Total Field Coverage: {total_pct:.1f}%")

        if active_areas == 0:
            rospy.loginfo("All areas covered!")
            rospy.loginfo("Stopping all drones...")
            # Ensure everyone gets a final stop command
            for controller in area_controllers:
                for drone in controller.drone_explorers:
                     drone.stop_motion()
            break

            total_pts += t
            
        total_pct = (total_covered / total_pts * 100.0) if total_pts > 0 else 0.0
        rospy.loginfo_throttle(10, f"[SYSTEM] Total Field Coverage: {total_pct:.1f}%")

        if active_areas == 0:
            rospy.loginfo("All areas covered!")
            break

        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
    
    # Summarize
    for explorer in explorers:
        if not explorer.exploration_complete:
            explorer.record_summary('aborted', 'Mission Timeout/Ending')

    rospy.sleep(1.0) # Grace time
    
    # Join threads
    for thread in threads:
        thread.join(timeout=1.0)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("[CHECK] MISSION COMPLETE!")
    
    # Reporting
    mission_results = aggregator.get_results()
    build_allocation_report(report_path, areas, area_profiles, allocation_counts, full_plan, mission_results)
    write_mission_summary(summary_path, areas, full_plan, mission_results)
    
    rospy.loginfo("All missions complete. Notifying fleet and shutting down in 5 seconds...")
    try:
        mission_pub.publish(Bool(data=True))
    except Exception:
        pass
    rospy.sleep(5.0)
    rospy.signal_shutdown("All missions completed")



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
