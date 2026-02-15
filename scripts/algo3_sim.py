
import math
import random
import copy

class GridWaypointManager:
    """
    Manages the discrete grid of waypoints.
    """
    def __init__(self, side_length, coverage_radius, origin_x, origin_y):
        self.side_length = side_length
        self.coverage_radius = coverage_radius
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        # Grid settings - 1.5m spacing
        self.spacing = 1.5 
        
        # Generate waypoints
        self.waypoints = []
        rows = int(side_length / self.spacing) + 1
        cols = int(side_length / self.spacing) + 1
        
        center_x = side_length / 2.0
        center_y = side_length / 2.0

        for r in range(rows):
            y = min(r * self.spacing, side_length - 0.5)
            for c in range(cols):
                x = min(c * self.spacing, side_length - 0.5)
                
                # Check if point is within circular radius
                dist = math.hypot(x - center_x, y - center_y)
                if dist <= coverage_radius:
                    self.waypoints.append((x, y))
            
        self.total_points = len(self.waypoints)
        self.visited = [False] * self.total_points
        self.visited_by = [-1] * self.total_points
        
    def get_unvisited_indices(self):
        return [i for i, v in enumerate(self.visited) if not v]

    def mark_visited(self, wp_idx, drone_id):
        if 0 <= wp_idx < self.total_points:
            if not self.visited[wp_idx]:
                self.visited[wp_idx] = True
                self.visited_by[wp_idx] = drone_id
                return True
        return False

    def get_progress_stats(self):
        visited_count = sum(self.visited)
        pct = (visited_count / self.total_points) * 100.0 if self.total_points > 0 else 100.0
        return visited_count, self.total_points, pct
        
    def get_drone_stats(self):
        stats = {}
        for d_id in self.visited_by:
            if d_id != -1:
                stats[d_id] = stats.get(d_id, 0) + 1
        return stats

    def find_closest_waypoint(self, x, y):
        best_idx = -1
        best_dist = float('inf')
        for i, (wx, wy) in enumerate(self.waypoints):
            dist = math.hypot(x - wx, y - wy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx


class Snapshot:
    def __init__(self, drone_targets, drone_positions, visited_mask):
        self.drone_targets = copy.deepcopy(drone_targets)
        self.drone_positions = copy.deepcopy(drone_positions)
        self.visited_mask = list(visited_mask) 

    def restore(self, algo):
        algo.drone_targets = copy.deepcopy(self.drone_targets)
        algo.drone_positions = copy.deepcopy(self.drone_positions)
        algo.manager.visited = list(self.visited_mask)


class Algo3Hybrid:
    """
    Strict implementation of Algo 3 (GOMWC + LERR + Shake + Revert).
    With SPATIAL PARTITIONING to ensure separate coverage zones.
    """
    def __init__(self, manager: GridWaypointManager, num_drones: int):
        self.manager = manager
        self.num_drones = num_drones
        
        # State
        self.drone_targets = {i: -1 for i in range(num_drones)}
        self.drone_positions = {i: (0,0) for i in range(num_drones)}

        # Algo Parameters
        self.shake_trigger_M = 4      
        self.delta = 4.0        
        self.epsilon = 0.1            
        
        self.no_gain_counter = 0
        self.best_coverage = 0.0
        self.shake_count = 0
        
        # Define Partitions (Vertical Strips)
        # e.g. 2 Drones: [0, W/2], [W/2, W]
        self.partition_width = self.manager.side_length / max(1, num_drones)

    def _is_in_partition(self, drone_id, wp_idx):
        """Check if waypoint belongs to drone's assigned strip."""
        if self.num_drones == 1: return True
        
        wx, wy = self.manager.waypoints[wp_idx]
        
        # X-based partitioning
        min_x = drone_id * self.partition_width
        max_x = (drone_id + 1) * self.partition_width
        
        # Include boundary for last drone to avoid rounding gaps
        if drone_id == self.num_drones - 1:
            max_x = self.manager.side_length + 1.0
            
        return min_x <= wx < max_x

    def update_drone_pose(self, drone_id, x, y):
        self.drone_positions[drone_id] = (x, y)
        
        # Check if reached current target
        tgt_idx = self.drone_targets[drone_id]
        if tgt_idx != -1:
            tx, ty = self.manager.waypoints[tgt_idx]
            dist = math.hypot(x - tx, y - ty)
            if dist < 0.5:
                # Reached! Mark visited
                self.manager.mark_visited(tgt_idx, drone_id)
                # Clear target to allow new selection
                self.drone_targets[drone_id] = -1
            else:
                # If we have a target but haven't reached it, KEEP IT.
                # Do not re-evaluate to avoid "wiggling".
                # Unless we are stuck or conflict.
                pass

    def force_mark_visited(self, drone_id):
        """Force mark current target as visited (called after scanning pause)."""
        tgt_idx = self.drone_targets[drone_id]
        if tgt_idx != -1:
            self.manager.mark_visited(tgt_idx, drone_id)
            self.drone_targets[drone_id] = -1 # Clear target
            return True
        return False

    def _calculate_score(self, drone_id, wp_idx):
        """
        Score = -Distance - TargetRepulsion
        Returns -inf if outside partition.
        """
        # 0. Partition Check (Hard Constraint)
        if not self._is_in_partition(drone_id, wp_idx):
            return -float('inf')

        wx, wy = self.manager.waypoints[wp_idx]
        px, py = self.drone_positions[drone_id]
        
        dist = math.hypot(px - wx, py - wy)
        
        # 1. Greedy Score
        score = -dist 
        
        # 2. Target Repulsion (Still useful for intra-partition or edge cases)
        for other_id, other_pos in self.drone_positions.items():
            if other_id == drone_id: continue
            
            tgt_idx = self.drone_targets[other_id]
            if tgt_idx != -1:
                tx, ty = self.manager.waypoints[tgt_idx]
                d_tgt = math.hypot(wx - tx, wy - ty)
                if d_tgt < 4.0: 
                    score -= 100.0

        return score

    def _shake_single(self, d_id):
        """Force a single drone to pick a random target IN ITS PARTITION."""
        px, py = self.drone_positions[d_id]
        
        # Random vector
        angle = random.uniform(0, 2*math.pi)
        dist = random.uniform(1.0, self.delta)
        
        new_x = max(0, min(px + dist*math.cos(angle), self.manager.side_length))
        new_y = max(0, min(py + dist*math.sin(angle), self.manager.side_length))
        
        # Find closest point that is VALID (in partition)
        best_idx = self.manager.find_closest_waypoint(new_x, new_y)
        
        if best_idx != -1 and self._is_in_partition(d_id, best_idx):
            self.drone_targets[d_id] = best_idx
            return True
            
        # If random jump landed outside, try scanning for ANY valid point
        # (Fallback)
        unvisited = self.manager.get_unvisited_indices()
        my_candidates = [i for i in unvisited if self._is_in_partition(d_id, i)]
        if my_candidates:
            self.drone_targets[d_id] = random.choice(my_candidates)
            return True
            
        return False

    def _shake_all(self):
        for i in range(self.num_drones):
            self._shake_single(i)

    def step(self):
        # 0. Measure State
        current_cov_pts, total_pts, current_cov_pct = self.manager.get_progress_stats()
        
        if current_cov_pts == total_pts:
            return "Mission Complete"
            
        # ---------------------------------------------------------
        # PROXIMITY CHECK (Anti-Cluster Force)
        # ---------------------------------------------------------
        # With partitioning, this should happen less.
        # If blocked, just WAIT or re-plan, don't jump randomly.
        drones_list = list(range(self.num_drones))
        
        for i in range(len(drones_list)):
            d1 = drones_list[i]
            p1 = self.drone_positions[d1]
            
            for j in range(i+1, len(drones_list)):
                d2 = drones_list[j]
                p2 = self.drone_positions[d2]
                
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist < 3.0: # 3.0m Safety Bubble (Increased)
                    # Force D2 to drop target and re-evaluate
                    self.drone_targets[d2] = -1
                    return f"PROXIMITY RESET (Px: {dist:.1f}m)"

        # Check Progress
        if current_cov_pct > self.best_coverage + self.epsilon:
            self.best_coverage = current_cov_pct
            self.no_gain_counter = 0
        else:
            self.no_gain_counter += 1
            
        # ---------------------------------------------------------
        # SHAKE LOGIC (Stagnation) - RESTORED
        # ---------------------------------------------------------
        # "Shake I mentioned for relaxation is only when drones are very close and there's a path blockage."
        # Actually user said "Stagnation shake is necessary."
        if self.no_gain_counter >= self.shake_trigger_M:
            self.shake_count += 1
            self._shake_all()
            self.no_gain_counter = 0 
            return f"SHAKE ({self.shake_count})"

        # ---------------------------------------------------------
        # GOMWC LOGIC (Partitioned)
        # ---------------------------------------------------------
        unvisited = self.manager.get_unvisited_indices()
        
        for d_id in range(self.num_drones):
            # Target Locking: If already has target, keep it! 
            # This prevents "wiggling while moving".
            if self.drone_targets[d_id] != -1:
                continue 
                
            best_idx = -1
            best_score = -float('inf')
            
            # Optimization: Only check candidates in my partition
            # But iterating all unvisited is fine for N<100
            
            for idx in unvisited:
                score = self._calculate_score(d_id, idx)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                self.drone_targets[d_id] = best_idx
            else:
                # No valid points in my partition?
                # Check if my partition is done.
                # If I am done but others aren't, I should stay put or help?
                # "Separate areas" implies strict boundaries. I will hold position.
                pass
                
        return "GOMWC (Partitioned)"

    def get_target_coords(self, drone_id):
        idx = self.drone_targets[drone_id]
        if idx != -1:
            return self.manager.waypoints[idx]
        return None
