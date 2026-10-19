"""
Chapter 5: Point Cloud Registration & LiDAR Odometry
Code Example 5.4: Hessian Matrix Decoupling & Geometrical Degeneracy Detection.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第5章 5.3.1节
"""

import numpy as np
from typing import Dict, Any
import lidarlens as ll
from lidarlens.registration import detect_degeneracy


def simulate_tunnel_hessian() -> np.ndarray:
    """Simulate a 6x6 Gauss-Newton Hessian matrix H = J^T * J inside a long featureless tunnel.
    
    Inside a featureless cylindrical or box tunnel along X axis:
    - Lateral constraints (Y, Z) and roll/pitch/yaw are well constrained by walls and floor.
    - Longitudinal translation (X) is completely unconstrained (eigenvalue -> 0).
    """
    H = np.zeros((6, 6), dtype=np.float32)
    # Rotation block: well constrained
    H[:3, :3] = np.array([
        [450.0,  12.0,  -8.0],
        [ 12.0, 520.0,   5.0],
        [ -8.0,   5.0, 480.0]
    ], dtype=np.float32)
    
    # Translation block: X has almost zero constraint (~0.2), Y and Z are strong (~1200)
    H[3:, 3:] = np.array([
        [   0.15,   0.02,  -0.01],
        [   0.02, 1250.0,  14.0],
        [  -0.01,   14.0, 1180.0]
    ], dtype=np.float32)
    
    return H


def main():
    print("=== Chapter 5: Hessian Degeneracy Detection in Tunnel Scenarios ===")
    
    # 1. Normal well-constrained urban intersection scenario
    H_well_constrained = np.diag([500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0]).astype(np.float32)
    res_normal = detect_degeneracy(H_well_constrained, threshold=100.0)
    print(f"Normal Intersection: Is Degenerate = {res_normal['is_degenerate']}, "
          f"Translation Cond = {res_normal['condition_number_translation']:.2f}")
    assert not res_normal["is_degenerate"]
    
    # 2. Degenerate long featureless tunnel scenario
    H_tunnel = simulate_tunnel_hessian()
    res_tunnel = detect_degeneracy(H_tunnel, threshold=100.0)
    
    cond_rot = res_tunnel["condition_number_rotation"]
    cond_trans = res_tunnel["condition_number_translation"]
    is_deg = res_tunnel["is_degenerate"]
    deg_dirs = res_tunnel.get("degenerate_directions", [])
    
    print("\n" + "=" * 50)
    print("      Degeneracy Analysis in Feature-Poor Tunnel")
    print("=" * 50)
    print(f"Rotation Condition Number:    {cond_rot:.2f}")
    print(f"Translation Condition Number: {cond_trans:.2f} (Threshold=100.0)")
    print(f"System State:                 {'DEGENERATE' if is_deg else 'STABLE'}")
    print(f"Degenerate Directions Count:  {len(deg_dirs)}")
    if len(deg_dirs) > 0:
        v_deg = deg_dirs[0]
        print(f"Primary Degenerate Eigenvector: [{v_deg[0]:.4f}, {v_deg[1]:.4f}, {v_deg[2]:.4f}]")
        # In tunnel along X axis, primary degenerate vector is aligned with X axis [1, 0, 0]
        print(f"Longitudinal X alignment:     {abs(v_deg[0]):.4f} (~1.0 indicates tunnel slip)")
    print("=" * 50)
    
    assert is_deg, "Tunnel scenario should trigger degeneracy alarm!"
    assert abs(deg_dirs[0][0]) > 0.95, "Primary degeneracy must align with longitudinal X axis!"
    print("Chapter 5.4 verification completed successfully.\n")


if __name__ == "__main__":
    main()
