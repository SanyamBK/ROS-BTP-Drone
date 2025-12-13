#!/usr/bin/env python3

import rospy
import json
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

class SwarmLocalization:
    """
    Decentralized Localization using UWB Ranges.
    
    Consumes: /swarm/uwb_ranges (Simulated UWB hardware)
    Produces: /drone_X/local_pos (Relative position estimate)
    
    Logic:
    1. Listen to pairwise ranges.
    2. Maintain a local graph of neighbor distances.
    3. Use Multilateration (Least Squares) to estimate own position relative to anchors (0,0).
       (For simplicity in this phase, we assume Drone 0 is the anchor/origin).
    """
    
    def __init__(self):
        rospy.init_node('swarm_localization_node', anonymous=True)
        
        self.drone_id = rospy.get_param('~drone_id', 0) # Default to ID 0
        self.uwb_sub = rospy.Subscriber('/swarm/uwb_ranges', String, self.uwb_callback)
        self.pos_pub = rospy.Publisher('local_pos', Point, queue_size=10)
        
        self.known_ranges = {} # {neighbor_id: distance}
        self.estimated_pos = np.zeros(3)
        
        # In a real decentralized system, we need anchors. 
        # For simulation, we cheat slightly and subscribe to neighbor Odoms just to get anchor positions
        # OR we implement the full cooperative belief space optimization (from IROS paper).
        # For Step 1 (InfoCom), we just do basic trilateration assuming known neighbors.
        self.neighbor_approx_pos = {} 

        rospy.loginfo(f"[SwarmLoc] Localizer started for Drone {self.drone_id}")

    def uwb_callback(self, msg):
        try:
            data = json.loads(msg.data)
            # Filter for ranges involving ME
            my_ranges = [d for d in data if d['a'] == self.drone_id or d['b'] == self.drone_id]
            
            for r in my_ranges:
                neighbor = r['b'] if r['a'] == self.drone_id else r['a']
                dist = r['dist']
                self.known_ranges[neighbor] = dist
                
            self.compute_position()
            
        except ValueError:
            pass

    def compute_position(self):
        """
        Multilateration: Estimate position (x,y,z) given distances to anchors.
        Minimizes sum of squared errors: sum( (dist_i - ||pos - anchor_i||)^2 )
        """
        if len(self.known_ranges) < 3:
            return # Need at least 3 neighbors for 2D/3D fix
            
        # Get anchors (neighbors with known positions)
        # Note: In a real decentralized swarms, this is harder (Cooperative Localization).
        # Here we assume we can subscribe to their public 'local_pos' or 'odom'.
        # For this prototype, we'll mock anchors as randomly around 50m to show the math works.
        
        anchors = []
        distances = []
        
        for drone_id, dist in self.known_ranges.items():
            # In real impl: self.get_neighbor_pos(drone_id)
            # Mocking anchor position based on ID for demo stability
            # Drone 0 is at (0,0), Drone 1 at (10,0)... just for the solver to work validly
            # In Phase 3 we connect this to real neighbor odoms.
            mock_x = (drone_id % 4) * 20.0
            mock_y = (drone_id // 4) * 20.0
            anchors.append([mock_x, mock_y, 0.0])
            distances.append(dist)
            
        anchors = np.array(anchors)
        distances = np.array(distances)
        
        # Initial guess
        x0 = np.mean(anchors, axis=0)
        
        def residuals(x, anchors, dists):
            return np.linalg.norm(anchors - x, axis=1) - dists
            
        try:
            from scipy.optimize import least_squares
            res = least_squares(residuals, x0, args=(anchors, distances))
            
            p = Point()
            p.x = res.x[0]
            p.y = res.x[1]
            p.z = res.x[2]
            self.pos_pub.publish(p)
            
            # --- BELIEF SPACE METRIC (IROS 2024) ---
            # Estimate Covariance Sigma = inverse(J.T * J) * MSE
            # J = Jacobian at the solution
            J = res.jac
            mse = np.mean(res.fun**2)
            try:
                # covariance matrix approximation
                cov = np.linalg.inv(J.T @ J) * mse
                belief_uncertainty = np.trace(cov) # minimization objective J = tr(Sigma)
            except np.linalg.LinAlgError:
                belief_uncertainty = 999.0

            # VISIBILITY UPDATE: Log fix and Belief Uncertainty
            # Calculate simple velocity (dist / time) since last fix or use odom if avail
            # For this log, we'll use a placeholder or derived value
            vel_mag = 0.0
            if hasattr(self, 'last_pos') and hasattr(self, 'last_time'):
                dt = (rospy.Time.now() - self.last_time).to_sec()
                if dt > 0:
                    dist = np.linalg.norm(np.array([p.x, p.y, p.z]) - self.last_pos)
                    vel_mag = dist / dt
            
            self.last_pos = np.array([p.x, p.y, p.z])
            self.last_time = rospy.Time.now()

            rospy.loginfo_throttle(5.0, 
                f"[SwarmLoc] Time: {rospy.Time.now().to_sec():.2f} | Drone {self.drone_id} | "
                f"Pos: ({p.x:.1f}, {p.y:.1f}) | Vel: {vel_mag:.2f} m/s | "
                f"Belief Uncertainty (tr(Sigma)): {belief_uncertainty:.4f}"
            )
            
            error = np.mean(np.abs(res.fun))
            if error < 0.5:
                pass # Good fix
                
        except ImportError:
            rospy.logerr("scipy not installed, cannot solve localization")

if __name__ == '__main__':
    try:
        node = SwarmLocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
