"""
Chapter 4: Complex Terrain Feature Extraction & Obstacle Clustering
Code Example 4.2: Ground Surface Segmentation (RANSAC & Patchwork++ CZM).

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第4章 4.3节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.filters.ground import RANSACGround, PatchworkPlusPlus


def generate_sloped_road_with_obstacles(num_points: int = 50_000) -> np.ndarray:
    """Generate urban street point cloud with sloped ground, curbs, cars, and trees."""
    pts = np.zeros((num_points, 4), dtype=np.float32)
    # Ground plane with 3-degree slope
    n_ground = int(num_points * 0.70)
    pts[:n_ground, 0] = np.random.uniform(0.0, 50.0, n_ground)
    pts[:n_ground, 1] = np.random.uniform(-15.0, 15.0, n_ground)
    # Slope equation: z = -0.05 * x + noise
    pts[:n_ground, 2] = -0.05 * pts[:n_ground, 0] - 1.6 + np.random.normal(0, 0.03, n_ground)
    pts[:n_ground, 3] = np.random.uniform(0.1, 0.5, n_ground)
    
    # Obstacles: vehicles and pedestrians
    n_obs = num_points - n_ground
    idx_obs = slice(n_ground, num_points)
    pts[idx_obs, 0] = np.random.uniform(5.0, 45.0, n_obs)
    pts[idx_obs, 1] = np.random.uniform(-10.0, 10.0, n_obs)
    pts[idx_obs, 2] = -0.05 * pts[idx_obs, 0] - 1.6 + np.random.uniform(0.2, 2.5, n_obs)
    pts[idx_obs, 3] = np.random.uniform(0.5, 0.95, n_obs)
    return pts


def main():
    print("=== Chapter 4: Ground Segmentation (RANSAC vs Patchwork++) ===")
    
    cloud = generate_sloped_road_with_obstacles(num_points=50_000)
    frame = ll.LidarFrame(points=cloud)
    print(f"Total raw points: {len(cloud):,}")
    
    # 1. RANSAC Planar Ground Segmentation
    ransac = RANSACGround(distance_threshold=0.15, max_iterations=50)
    t0 = time.perf_counter()
    ground_mask_ransac, plane_model = ransac.segment(frame)
    ransac_ms = (time.perf_counter() - t0) * 1000.0
    ground_pts_ransac = cloud[ground_mask_ransac]
    non_ground_pts_ransac = cloud[~ground_mask_ransac]
    print(f"RANSAC Ground Segmentation in {ransac_ms:.2f} ms: "
          f"Ground={len(ground_pts_ransac):,}, Non-Ground={len(non_ground_pts_ransac):,}, "
          f"Plane=[{plane_model[0]:.2f}, {plane_model[1]:.2f}, {plane_model[2]:.2f}, {plane_model[3]:.2f}]")
    
    # 2. Patchwork++ Concentric Zone Model (CZM) Ground Segmentation
    patchwork = PatchworkPlusPlus(czm_rings=4, czm_sectors=16, elevation_thresh=0.20)
    t0 = time.perf_counter()
    ground_pw_mask, non_ground_pw_mask = patchwork.segment(frame)
    pw_ms = (time.perf_counter() - t0) * 1000.0
    ground_pw = cloud[ground_pw_mask]
    non_ground_pw = cloud[non_ground_pw_mask]
    print(f"Patchwork++ CZM Segmentation in {pw_ms:.2f} ms: "
          f"Ground={len(ground_pw):,}, Non-Ground={len(non_ground_pw):,}")
    
    assert len(ground_pw) > 0 and len(non_ground_pw) > 0
    assert len(ground_pts_ransac) > 0 and len(non_ground_pts_ransac) > 0
    print("Chapter 4.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
