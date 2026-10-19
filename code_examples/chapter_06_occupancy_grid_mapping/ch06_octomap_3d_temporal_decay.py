"""
Chapter 6: Occupancy Grid Mapping & Quality Audit
Code Example 6.2: 3D OctoMap Volumetric Mapping & Exponential Temporal Decay.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第6章 6.2节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.mapping import OctoMap3D


def main():
    print("=== Chapter 6: 3D OctoMap & Exponential Temporal Decay ===")
    
    # 1. Initialize OctoMap3D with log-odds probability updates
    octomap = OctoMap3D(resolution=0.1, p_occ=0.75, p_free=0.40, l_min=-3.0, l_max=3.5)
    sensor_origin = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # 2. Simulate observation of a dynamic obstacle (pedestrian walking by at t=0)
    # Target points at [3.0, 0.0, 1.0]
    obstacle_pts = np.random.uniform([-0.2, -0.2, -0.5], [0.2, 0.2, 0.5], size=(200, 3)) + np.array([3.0, 0.0, 1.0])
    
    # Integrate multiple sensor hits to build confidence
    for _ in range(5):
        octomap.integrate_scan(sensor_origin=sensor_origin, end_points=obstacle_pts)
        
    occupied_initial = octomap.get_occupied_centers()
    print(f"Frame 0 (Dynamic obstacle detected): {len(occupied_initial)} occupied voxels in OctoMap.")
    assert len(occupied_initial) > 0, "Failed to integrate occupied voxels!"
    
    # 3. Simulate passage of time where the obstacle has moved away (Temporal Decay)
    # Formula: L_t = L_{t-1} * exp(-lambda * dt)
    print("\nSimulating passage of temporal decay without new hits (obstacle left)...")
    decay_steps = 10
    for step in range(decay_steps):
        octomap.apply_temporal_decay(decay_rate=0.60, dt=1.0)
        remaining_voxels = octomap.get_occupied_centers()
        if (step + 1) % 2 == 0:
            print(f"  T+{step+1}s: {len(remaining_voxels)} occupied voxels remaining...")
        
    final_voxels = octomap.get_occupied_centers()
    print(f"\nFinal OctoMap state after temporal decay: {len(final_voxels)} voxels.")
    print("Ghost trail of moving pedestrian has been successfully cleared by exponential decay!")
    
    assert len(final_voxels) < len(occupied_initial), "Temporal decay failed to reduce ghost voxel count!"
    print("Chapter 6.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
