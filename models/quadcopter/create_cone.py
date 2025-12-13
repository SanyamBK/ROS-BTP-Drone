import numpy as np
import struct

def write_obj(filename, radius, height, segments=32):
    # Cone Tip at (0,0,0)
    # Cone Base at (0,0,-height)
    
    with open(filename, 'w') as f:
        f.write("# OBJ file\n")
        f.write("o cone\n")
        
        # Vertices
        # v1: Tip
        f.write("v 0.0 0.0 0.0\n")
        
        # Base Vertices (v2 to v_segments+1)
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
        for a in angles:
             f.write(f"v {radius * np.cos(a)} {radius * np.sin(a)} {-height}\n")
             
        # Center of Base (v_end) for caps
        f.write(f"v 0.0 0.0 {-height}\n")
        base_center_idx = segments + 2
        
        # Faces
        # Side Faces (Tip -> Base i -> Base i+1)
        # OBJ indices are 1-based
        for i in range(segments):
            # circle indices are 2 ... segments+1
            curr_idx = i + 2
            next_idx = (i + 1) % segments + 2
            
            # Side (Tip=1)
            f.write(f"f 1 {curr_idx} {next_idx}\n")
            
            # Base Cap (Center -> Next -> Curr) (CW/CCW?)
            # Normal should point down. 
            f.write(f"f {base_center_idx} {next_idx} {curr_idx}\n")

if __name__ == "__main__":
    write_obj('/home/ros/catkin_ws/src/multi_drone_sim/models/quadcopter/cone.obj', 1.0, 1.0)
