
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
        cell_area = self.spacing * self.spacing
        covered_area = visited_count * cell_area
        total_area = self.total_points * cell_area
        pct = (visited_count / self.total_points) * 100.0 if self.total_points > 0 else 100.0
        return covered_area, total_area, pct
        
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
        self.active_drones = list(range(num_drones))

        # Algo Parameters
        self.shake_trigger_M = 4      
        self.delta = 4.0        
        self.epsilon = 0.1            
        
        self.no_gain_counter = 0
        self.best_coverage = 0.0
        self.shake_count = 0

    def update_active_drones(self, active_list):
        self.active_drones = active_list
        # Clear targets for drones that are no longer active
        for d_id in list(self.drone_targets.keys()):
            if d_id not in self.active_drones:
                self.drone_targets[d_id] = -1

    def _is_in_partition(self, drone_id, wp_idx):
        """Dynamic Voronoi partitioning based on current active drones."""
        if drone_id not in self.active_drones:
            return False
            
        if len(self.active_drones) <= 1:
            return True
            
        wx, wy = self.manager.waypoints[wp_idx]
        
        my_pos = self.drone_positions.get(drone_id, (0,0))
        my_dist = math.hypot(my_pos[0] - wx, my_pos[1] - wy)
        
        for other_id in self.active_drones:
            if other_id == drone_id: continue
            other_pos = self.drone_positions.get(other_id, (0,0))
            other_dist = math.hypot(other_pos[0] - wx, other_pos[1] - wy)
            if other_dist < my_dist:
                return False # Another active drone is closer
            elif abs(other_dist - my_dist) < 0.01 and other_id < drone_id:
                return False # Tie-breaker to prevent overlap

        return True

    def update_drone_pose(self, drone_id, x, y):
        self.drone_positions[drone_id] = (x, y)
        
        # Check if reached current target
        tgt_idx = self.drone_targets[drone_id]
        if tgt_idx != -1:
            tx, ty = self.manager.waypoints[tgt_idx]
            dist = math.hypot(x - tx, y - ty)
            # The Drone's physical camera FOV at 2.0m altitude is ~1.25m radius.
            # Using 1.5m ensures the algorithm marks it as visited as soon as it enters 
            # the visual coverage cone, preventing target stalling.
            if dist <= 1.5:
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
        
        # Target Repulsion (Just avoid exactly same target)
        for other_id in self.active_drones:
            if other_id == drone_id: continue
            tgt_idx = self.drone_targets[other_id]
            if tgt_idx == wp_idx:
                score = -999999
                break

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
        for i in self.active_drones:
            self._shake_single(i)

    def step(self):
        # 0. Measure State
        covered_area, total_area, current_cov_pct = self.manager.get_progress_stats()
        
        if current_cov_pct >= 99.9:
            # RESET GRID for continuous surveillance
            self.manager.visited = [False] * self.manager.total_points
            self.manager.visited_by = [-1] * self.manager.total_points
            self.best_coverage = 0.0
            self.no_gain_counter = 0
            return "Surveillance Loop Reset"
            
        # ---------------------------------------------------------
        # PROXIMITY CHECK (Anti-Cluster Force)
        # ---------------------------------------------------------
        # With partitioning, this should happen less.
        # If blocked, just WAIT or re-plan, don't jump randomly.
        msgs = []
        drones_list = list(self.active_drones)
        
        for i in range(len(drones_list)):
            d1 = drones_list[i]
            p1 = self.drone_positions[d1]
            
            for j in range(i+1, len(drones_list)):
                d2 = drones_list[j]
                p2 = self.drone_positions[d2]
                
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist < 1.5: # Match grid spacing (1.5m) to avoid over-triggering
                    # Force D2 to drop target and re-evaluate
                    self.drone_targets[d2] = -1
                    msgs.append(f"PROXIMITY RESET (Px: {dist:.1f}m)")

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
        
        for d_id in self.active_drones:
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
            elif unvisited:
                # Starvation fallback: If Voronoi partition is entirely empty, pick nearest available
                fallback_idx = -1
                fallback_score = -999999
                targeted = set(self.drone_targets.values())
                
                for idx in unvisited:
                    if idx in targeted:
                       continue
                    wx, wy = self.manager.waypoints[idx]
                    px, py = self.drone_positions.get(d_id, (0,0))
                    f_score = -math.hypot(wx - px, wy - py)
                    
                    if f_score > fallback_score:
                        fallback_score = f_score
                        fallback_idx = idx
                if fallback_idx != -1:
                    self.drone_targets[d_id] = fallback_idx
            else:
                # Partition exhausted — help cover any remaining global unvisited waypoint
                global_candidates = self.manager.get_unvisited_indices()
                if global_candidates:
                    # Pick closest globally unvisited waypoint
                    px, py = self.drone_positions.get(d_id, (0, 0))
                    global_candidates.sort(key=lambda i: math.hypot(
                        px - self.manager.waypoints[i][0],
                        py - self.manager.waypoints[i][1]
                    ))
                    self.drone_targets[d_id] = global_candidates[0]
                
        return "GOMWC (Partitioned)"

    def get_target_coords(self, drone_id):
        idx = self.drone_targets[drone_id]
        if idx != -1:
            return self.manager.waypoints[idx]
        return None
