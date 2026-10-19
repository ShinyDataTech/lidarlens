"""
Chapter 4: Complex Terrain Feature Extraction & Obstacle Clustering
Code Example 4.1: Vectorized 2.5D Elevation Mapping & Traversability Analysis.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第4章 4.1 & 4.2节
"""

import time
import numpy as np
from typing import Dict, Any
import lidarlens as ll
from lidarlens.terrain import create_elevation_map, analyze_traversability


def generate_rugged_terrain_cloud(num_points: int = 60_000) -> np.ndarray:
    """Generate 3D point cloud of uneven terrain with ramps, rocks, and obstacles."""
    pts = np.zeros((num_points, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-10.0, 10.0, num_points)
    pts[:, 1] = np.random.uniform(-10.0, 10.0, num_points)
    
    # Base undulating terrain (sine wave slope)
    pts[:, 2] = 0.3 * np.sin(0.4 * pts[:, 0]) + 0.15 * np.cos(0.5 * pts[:, 1])
    
    # Add a steep ramp (slope > 0.3)
    ramp_mask = (pts[:, 0] > 2.0) & (pts[:, 0] < 6.0) & (pts[:, 1] > -3.0) & (pts[:, 1] < 3.0)
    pts[ramp_mask, 2] += 0.8 * (pts[ramp_mask, 0] - 2.0)
    
    # Add rock / step obstacle (step height > 0.35m)
    rock_mask = (pts[:, 0] > -5.0) & (pts[:, 0] < -3.0) & (pts[:, 1] > 1.0) & (pts[:, 1] < 3.0)
    pts[rock_mask, 2] += 0.5
    
    # Ground roughness noise
    pts[:, 2] += np.random.normal(0, 0.02, num_points)
    pts[:, 3] = np.random.uniform(0.1, 0.8, num_points)
    return pts


def main():
    print("=== Chapter 4: Vectorized Elevation Mapping & Traversability Analysis ===")
    
    # 1. Generate rugged terrain point cloud
    cloud = generate_rugged_terrain_cloud(num_points=60_000)
    print(f"Generated rugged terrain cloud: {len(cloud):,} points.")
    
    # 2. Vectorized Elevation Map Generation (<5 ms target)
    t0 = time.perf_counter()
    z_map, z_max, z_min = create_elevation_map(
        points=cloud,
        grid_size=0.1,
        x_range=(-10.0, 10.0),
        y_range=(-10.0, 10.0)
    )
    elev_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Elevation map generated in {elev_ms:.2f} ms (Grid shape: {z_map.shape}).")
    assert elev_ms < 20.0, f"Elevation mapping took {elev_ms:.2f} ms, expected < 20 ms"
    
    # 3. Traversability Costmap Analysis
    t0 = time.perf_counter()
    costmap, traversable_mask = analyze_traversability(
        z_map=z_map,
        z_max=z_max,
        z_min=z_min,
        res=0.1
    )
    analysis_ms = (time.perf_counter() - t0) * 1000.0
    
    valid_cells = ~np.isnan(z_map)
    total_valid = np.sum(valid_cells)
    traversable_cells = np.sum(traversable_mask > 0)
    traversable_ratio = traversable_cells / max(total_valid, 1) * 100.0
    
    print(f"Traversability analysis completed in {analysis_ms:.2f} ms.")
    print(f"Traversable Area Ratio: {traversable_ratio:.1f}%")
    print(f"Costmap value bounds:   min={costmap.min():.4f}, max={costmap.max():.4f}")
    
    assert costmap.shape == z_map.shape
    assert traversable_mask.shape == z_map.shape
    print("Chapter 4.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
