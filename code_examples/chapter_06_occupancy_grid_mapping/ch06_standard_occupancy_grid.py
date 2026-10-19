"""
Chapter 6: Occupancy Grid Mapping & Quality Audit
Code Example 6.1: Standard ROS 2 OccupancyGrid Generation & Safety Inflation.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第6章 6.3.1节
"""

import time
import numpy as np
from scipy.ndimage import binary_dilation
import lidarlens as ll
from lidarlens.mapping import generate_standard_grid


def generate_room_scan(num_points: int = 40_000) -> np.ndarray:
    """Generate 3D indoor room scan with perimeter walls, columns, and tables."""
    pts = []
    # 1. Four boundary walls (10m x 8m room)
    n_wall = num_points // 4
    x_w = np.random.uniform(-5.0, 5.0, n_wall)
    z_w = np.random.uniform(0.0, 2.5, n_wall)
    pts.append(np.column_stack([x_w, np.full(n_wall, -4.0), z_w]))
    pts.append(np.column_stack([x_w, np.full(n_wall, 4.0), z_w]))
    
    y_w = np.random.uniform(-4.0, 4.0, n_wall)
    pts.append(np.column_stack([np.full(n_wall, -5.0), y_w, z_w]))
    pts.append(np.column_stack([np.full(n_wall, 5.0), y_w, z_w]))
    
    # 2. Interior columns (radius ~0.4m at [1.5, 1.5] and [-1.5, -1.5])
    n_col = 1500
    th = np.random.uniform(0, 2*np.pi, n_col)
    z_c = np.random.uniform(0, 2.0, n_col)
    pts.append(np.column_stack([1.5 + 0.4*np.cos(th), 1.5 + 0.4*np.sin(th), z_c]))
    pts.append(np.column_stack([-1.5 + 0.4*np.cos(th), -1.5 + 0.4*np.sin(th), z_c]))
    
    # 3. Floor ground points (z in [-0.05, 0.05])
    n_floor = 10_000
    pts.append(np.column_stack([
        np.random.uniform(-4.8, 4.8, n_floor),
        np.random.uniform(-3.8, 3.8, n_floor),
        np.random.normal(0.0, 0.02, n_floor)
    ]))
    
    cloud = np.vstack(pts).astype(np.float32)
    # Add dummy intensity
    xyzi = np.column_stack([cloud, np.random.uniform(0.2, 0.8, len(cloud))]).astype(np.float32)
    return xyzi


def main():
    print("=== Chapter 6: ROS 2 Standard OccupancyGrid Generation ===")
    
    cloud = generate_room_scan(num_points=35_000)
    print(f"Loaded room point cloud: {len(cloud):,} points.")
    
    # 1. Generate ROS 2 Occupancy Grid
    # Height slice [0.1m, 2.0m] filters out the floor ground while capturing obstacles
    t0 = time.perf_counter()
    inflated_grid, meta = generate_standard_grid(
        points=cloud,
        resolution=0.05,
        z_range=(0.10, 2.0),
        inflation_radius=0.15
    )
    grid_ms = (time.perf_counter() - t0) * 1000.0
    
    n_free = np.sum(inflated_grid == 0)
    n_lethal = np.sum(inflated_grid == 100)
    n_inflated = np.sum((inflated_grid > 0) & (inflated_grid < 100))
    n_unknown = np.sum(inflated_grid == -1)
    
    print("\n" + "=" * 50)
    print("        ROS 2 Standard Costmap Rasterization")
    print("=" * 50)
    print(f"Rasterization Time:   {grid_ms:.2f} ms")
    print(f"Grid Dimensions:      {inflated_grid.shape[1]} x {inflated_grid.shape[0]} (res={meta['resolution']}m)")
    print(f"Origin Coordinates:   ({meta['origin_x']:.2f}, {meta['origin_y']:.2f})")
    print(f"Free Cells (0):       {n_free:,} ({n_free/inflated_grid.size*100:.1f}%)")
    print(f"Lethal Cells (100):   {n_lethal:,} ({n_lethal/inflated_grid.size*100:.1f}%)")
    print(f"Inflated Margin:      {n_inflated:,} cells")
    print(f"Unknown Cells (-1):   {n_unknown:,}")
    print("=" * 50)
    
    assert n_lethal > 0, "No lethal obstacle cells found!"
    assert n_free > 0, "No free space cells found!"
    assert grid_ms < 50.0, f"Grid generation {grid_ms:.2f} ms exceeds latency budget!"
    print("Chapter 6.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
