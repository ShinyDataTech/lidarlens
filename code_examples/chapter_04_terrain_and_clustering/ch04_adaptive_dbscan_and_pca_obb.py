"""
Chapter 4: Complex Terrain Feature Extraction & Obstacle Clustering
Code Example 4.3: Range-Adaptive DBSCAN Clustering & 2.5D PCA OBB Extraction.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第4章 4.4节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.cluster import AdaptiveDBSCAN
from lidarlens.geometry import get_pca_obb, project_to_costmap


def generate_multicluster_obstacles() -> np.ndarray:
    """Generate multiple synthetic obstacles (cars, poles, boxes) at different distances."""
    clusters = []
    
    # 1. Close small obstacle (trash bin / box at 6m)
    c1 = np.random.uniform(-0.4, 0.4, size=(200, 3)) + np.array([6.0, 1.5, 0.0])
    clusters.append(c1)
    
    # 2. Medium distance vehicle (car at 18m, rotated ~30 degrees)
    c2 = np.random.uniform([-2.2, -0.9, -0.5], [2.2, 0.9, 0.9], size=(800, 3))
    rot_angle = np.deg2rad(30)
    R = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle),  np.cos(rot_angle), 0],
        [0, 0, 1]
    ])
    c2_rot = (R @ c2.T).T + np.array([18.0, -3.0, 0.5])
    clusters.append(c2_rot)
    
    # 3. Distant vehicle (sparse car at 40m)
    c3 = np.random.uniform([-2.2, -0.9, -0.5], [2.2, 0.9, 0.9], size=(250, 3)) + np.array([40.0, 2.0, 0.5])
    clusters.append(c3)
    
    all_pts = np.vstack(clusters).astype(np.float32)
    # Append scalar intensity
    all_xyzi = np.column_stack([all_pts, np.random.uniform(0.3, 0.9, len(all_pts))]).astype(np.float32)
    return all_xyzi


def main():
    print("=== Chapter 4: Adaptive DBSCAN & PCA OBB Extraction ===")
    
    obstacles = generate_multicluster_obstacles()
    print(f"Total obstacle candidate points: {len(obstacles):,}")
    
    # 1. Range-Adaptive DBSCAN Clustering
    clusterer = AdaptiveDBSCAN(eps_min=0.35, alpha=0.012, min_samples=15)
    t0 = time.perf_counter()
    labels = clusterer.fit_predict(obstacles)
    cluster_ms = (time.perf_counter() - t0) * 1000.0
    
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    print(f"Adaptive DBSCAN completed in {cluster_ms:.2f} ms: Found {len(unique_labels)} clusters.")
    assert len(unique_labels) >= 2, f"Expected at least 2 clusters, got {len(unique_labels)}"
    
    # 2. Extract 2.5D PCA OBB for each cluster
    obbs = []
    for cid in unique_labels:
        cluster_pts = obstacles[labels == cid]
        center, size, angle = get_pca_obb(cluster_pts)
        obbs.append((center, size, angle))
        yaw_deg = np.rad2deg(angle)
        print(f"Cluster #{cid} ({len(cluster_pts)} pts): "
              f"Center=[{center[0]:.2f}, {center[1]:.2f}], "
              f"Extent=[L={size[0]:.2f}, W={size[1]:.2f}], "
              f"Yaw={yaw_deg:.1f}°")
        
    # 3. Project OBB to BEV Costmap Grid
    grid_shape = (200, 200) # 20m x 20m at 0.1m resolution
    costmap = np.zeros(grid_shape, dtype=np.uint8)
    for center, size, angle in obbs:
        costmap = project_to_costmap(
            center=center,
            size=size,
            angle=angle,
            map_img=costmap,
            res=0.1
        )
        
    occupied_cells = np.sum(costmap > 0)
    print(f"Projected {len(obbs)} OBBs to costmap: {occupied_cells} occupied cells.")
    assert occupied_cells > 0
    print("Chapter 4.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
