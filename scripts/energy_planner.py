#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from math import sqrt, atan2
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState

import heapq
from math import floor

class EnergyAwarePlanner:
    """
    Coordinated Planner for Energy-Constrained Drones (ICRA 2024).
    Implements Priority-Based Path Planning for UGV.
    NOW FEATURING: Dijkstra Grid Search!
    """
    
    def __init__(self):
        rospy.init_node('energy_planner', anonymous=True)
        
        # Link to Gazebo
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        self.num_drones = rospy.get_param('~num_drones', 11)
        self.drone_states = {} 
        self.ugv_ids = ['ugv1', 'ugv2']
        self.ugvs = {}
        for uid in self.ugv_ids:
            self.ugvs[uid] = {
                'pos': Point(0,0,0),
                'yaw': 0.0,
                'state': 'IDLE',
                'target_id': None,
                'path': [],
                'path_idx': 0,
                'last_charge_time': rospy.Time.now(),
                'cmd_pub': rospy.Publisher(f'/{uid}/cmd_vel', Twist, queue_size=10)
            }
            rospy.Subscriber(f'/{uid}/odom', Odometry, self.ugv_cb, callback_args=uid)

        # Planning State
        # self.path = [] # Removed single path
        # self.path_idx = 0
        
        # Subscribe to Drone Data
        for i in range(self.num_drones):
            self.drone_states[i] = {'bat': 100.0, 'pos': Point(), 'under_charge': False}
            rospy.Subscriber(f'/drone_{i}/battery', Float32, self.bat_cb, callback_args=i)
            rospy.Subscriber(f'/drone_{i}/odom', Odometry, self.odom_cb, callback_args=i)
            
        self.charge_pubs = {}
        for i in range(self.num_drones):
            self.charge_pubs[i] = rospy.Publisher(f'/drone_{i}/charge_cmd', Float32, queue_size=10)
            
        self.rate = rospy.Rate(10)
        
        # LOGGING USER SPECS
        print("\n" + "="*60)
        rospy.loginfo("[EnergyPlanner] INITIALIZED with User Logic:")
        rospy.loginfo(" > Method: Dijkstra Grid Search (2m resolution)")
        rospy.loginfo(" > Planner: Multi-UGV Priority Assignment")
        rospy.loginfo(" > Charging: Instant Charge after 1s dwell.")
        print("="*60 + "\n")

    def bat_cb(self, msg, drone_id):
        if self.drone_states[drone_id]['bat'] == 100.0 and msg.data < 100.0:
             rospy.loginfo_throttle(1, f"[Planner] Drone {drone_id} reported battery: {msg.data:.1f}%")
        self.drone_states[drone_id]['bat'] = msg.data

    def odom_cb(self, msg, drone_id):
        self.drone_states[drone_id]['pos'] = msg.pose.pose.position

    def ugv_cb(self, msg, ugv_id):
        self.ugvs[ugv_id]['pos'] = msg.pose.pose.position
        # Extract yaw from quaternion (simplified)
        q = msg.pose.pose.orientation
        self.ugvs[ugv_id]['yaw'] = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))



    def get_priority_score(self, drone_id, ugv_pos):
        data = self.drone_states[drone_id]
        if data['bat'] > 95.0: return 0.0 
        if data['under_charge']: return 0.0 # Already being handled?
        
        # GEOP FENCE: Ignore drones at spawn/home base (Y < -15.0)
        # The farm is roughly -12 to +12. Spawn is -20.
        if data['pos'].y < -15.0:
            return 0.0
        
        dx = data['pos'].x - ugv_pos.x
        dy = data['pos'].y - ugv_pos.y
        dist = sqrt(dx*dx + dy*dy)
        if dist < 0.1: dist = 0.1
        
        urgency = (100.0 - data['bat']) ** 2
        cost = dist + 1.0
        return urgency / cost

    # ================= DIJKSTRA IMPLEMENTATION =================
    def grid_from_world(self, x, y):
        """Convert world coords to grid index (2m res)."""
        # offset 50 to handle negative coords
        gx = int(floor((x + 50.0) / 2.0))
        gy = int(floor((y + 50.0) / 2.0))
        return (gx, gy)

    def world_from_grid(self, gx, gy):
        """Convert grid index to world coords."""
        x = (gx * 2.0) - 50.0 + 1.0 # center of cell
        y = (gy * 2.0) - 50.0 + 1.0
        return (x, y)

    def plan_path_dijkstra(self, start_pos, goal_pos):
        """Dijkstra Algorithm on a 50x50 Grid (100mx100m world)."""
        start_node = self.grid_from_world(start_pos.x, start_pos.y)
        goal_node = self.grid_from_world(goal_pos.x, goal_pos.y)
        
        rospy.loginfo(f"[Dijkstra] Planning {start_node} -> {goal_node}...")
        
        frontier = []
        heapq.heappush(frontier, (0, start_node))
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal_node:
                break
            
            # Neighbors (4-connected)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_node = (current[0]+dx, current[1]+dy)
                
                # Check bounds (0 to 50 for 100m width / 2m res)
                if not (0 <= next_node[0] <= 50 and 0 <= next_node[1] <= 50):
                    continue
                    
                new_cost = cost_so_far[current] + 1 # Uniform cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost # Dijkstra
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
        # Reconstruct Path
        if goal_node not in came_from:
            rospy.logwarn("[Dijkstra] No path found!")
            return []
            
        path = []
        curr = goal_node
        while curr != start_node:
            wx, wy = self.world_from_grid(curr[0], curr[1])
            path.append((wx, wy))
            curr = came_from[curr]
        path.reverse() # Start -> Goal
        
        rospy.loginfo(f"[Dijkstra] Path found! Length: {len(path)} nodes.")
        return path

    def navigate_ugv(self, ugv_id):
        ugv = self.ugvs[ugv_id]
        target_id = ugv['target_id']
        
        if target_id is None:
            return

        target_pos = self.drone_states[target_id]['pos']
        
        # Check Final Distance
        dx = target_pos.x - ugv['pos'].x
        dy = target_pos.y - ugv['pos'].y
        dist = sqrt(dx*dx + dy*dy)
        
        if dist < 2.0:
            ugv['state'] = "CHARGING"
            ugv['last_charge_time'] = rospy.Time.now()
            ugv['cmd_pub'].publish(Twist()) # Stop
            rospy.loginfo(f"[Planner] {ugv_id} Arrived at Drone {target_id}. Charging...")
            return

        # Path Planning / Following
        if not ugv['path']:
            ugv['path'] = self.plan_path_dijkstra(ugv['pos'], target_pos)
            ugv['path_idx'] = 0
            
        if not ugv['path']: # Still no path? Just drive direct (Fallback)
             # rospy.logwarn_throttle(5, f"{ugv_id} Dijkstra failed. Fallback to direct drive.")
             target_x, target_y = target_pos.x, target_pos.y
        else:
             # Follow Path
             if ugv['path_idx'] < len(ugv['path']):
                 target_x, target_y = ugv['path'][ugv['path_idx']]
                 # Distance to intermediate waypoint
                 w_dx = target_x - ugv['pos'].x
                 w_dy = target_y - ugv['pos'].y
                 if sqrt(w_dx*w_dx + w_dy*w_dy) < 1.0: # Reached waypoint
                     ugv['path_idx'] += 1
                     if ugv['path_idx'] >= len(ugv['path']):
                         # Last point, aim for actual drone
                         target_x, target_y = target_pos.x, target_pos.y
                     else:
                         # ADD missing assignment for this tick
                         target_x, target_y = ugv['path'][ugv['path_idx']]
             else:
                 target_x, target_y = target_pos.x, target_pos.y

        # Control to intermediate target (target_x, target_y)
        t_dx = target_x - ugv['pos'].x
        t_dy = target_y - ugv['pos'].y
        target_yaw = atan2(t_dy, t_dx)
        err_yaw = target_yaw - ugv['yaw']
        while err_yaw > 3.14159: err_yaw -= 2*3.14159
        while err_yaw < -3.14159: err_yaw += 2*3.14159
        
        cmd = Twist()
        if abs(err_yaw) > 0.5:
            cmd.angular.z = 1.0 * (1 if err_yaw > 0 else -1)
        else:
            cmd.angular.z = 1.5 * err_yaw
            cmd.linear.x = 2.0 
            
        ugv['cmd_pub'].publish(cmd)

    def run_logic(self):
        
        # Identify Critical Drones NOT already targeted by a UGV
        # 1. Gather Needs
        # 2. Iterate UGVs
        
        # Set of targeted drones
        targeted_drones = set()
        for uid, u in self.ugvs.items():
            if u['target_id'] is not None:
                targeted_drones.add(u['target_id'])
                
        for uid in self.ugv_ids:
            ugv = self.ugvs[uid]
            
            if ugv['state'] == "IDLE" or ugv['state'] == "MOVING":
                # Check for higher priority target?
                # For simplicity: If IDLE, look for target. 
                # If MOVING, stick to target unless it charges itself (unlikely).
                
                if ugv['state'] == "IDLE":
                    # Find best unassigned target
                    scores = []
                    for i in self.drone_states:
                        if i in targeted_drones: continue
                        s = self.get_priority_score(i, ugv['pos'])
                        scores.append((i, s, self.drone_states[i]['bat']))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    if scores:
                        best_id, best_score, best_bat = scores[0]
                        if best_score > 0.1 and best_bat < 90.0: # Threshold
                            rospy.loginfo(f"[Planner] {uid} ASSIGNED -> Drone {best_id} (Bat: {best_bat:.1f}%)")
                            ugv['target_id'] = best_id
                            ugv['state'] = "MOVING"
                            ugv['path'] = []
                            targeted_drones.add(best_id)
                
                # Execute Moving Logic
                if ugv['state'] == "MOVING":
                    # Check if target is still valid (e.g. didn't fly away or get charged elsehow)
                    # If target battery > 95, cancel
                    tid = ugv['target_id']
                    if self.drone_states[tid]['bat'] > 95.0:
                         rospy.loginfo(f"[Planner] {uid} Target {tid} Full. Aborting.")
                         ugv['target_id'] = None
                         ugv['state'] = "IDLE"
                         ugv['cmd_pub'].publish(Twist())
                    else:
                         self.navigate_ugv(uid)

            elif ugv['state'] == "CHARGING":
                if (rospy.Time.now() - ugv['last_charge_time']).to_sec() >= 1.0:
                    tid = ugv['target_id']
                    rospy.loginfo(f"[Planner] {uid} Charged D{tid}!")
                    self.charge_pubs[tid].publish(100.0)
                    ugv['target_id'] = None
                    ugv['state'] = "IDLE"
                    ugv['path'] = []

    def run(self):
        rospy.sleep(2.0)
        while not rospy.is_shutdown():
            self.run_logic()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        planner = EnergyAwarePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
