"""
Chapter 2: Python 3 Development Environment & LidarLens Architecture
Code Example 2.4: Reflection Intensity-Based Turbo Colormap Pseudo-Color Rendering.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第2章 2.3.2节
"""

import numpy as np
from typing import Optional
import lidarlens as ll
from lidarlens.vis.visualizer import render_intensity_pointcloud, HAS_OPEN3D


def generate_curved_corridor_cloud(n_points: int = 50_000) -> np.ndarray:
    """Generate synthetic curved road tunnel point cloud with varying intensity."""
    theta = np.random.uniform(0, np.pi, n_points)
    arc_length = np.random.uniform(0, 40, n_points)
    
    # Corridor cross-section: radius ~ 4m
    radius = 4.0 + np.random.normal(0, 0.05, n_points)
    x = arc_length
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    
    # Distance attenuation + retroreflective lane markings
    base_intensity = np.exp(-0.04 * x)
    retro_mask = (np.abs(y) < 0.2) | (np.abs(y - 2.0) < 0.15) | (np.abs(y + 2.0) < 0.15)
    base_intensity[retro_mask] = 0.95
    
    cloud = np.column_stack([x, y, z, base_intensity]).astype(np.float32)
    return cloud


def main():
    print("=== Chapter 2: Intensity-Based Turbo Colormap Rendering ===")
    
    # 1. Generate synthetic corridor
    cloud = generate_curved_corridor_cloud(50_000)
    print(f"Generated point cloud: shape={cloud.shape}, intensity range=[{cloud[:, 3].min():.2f}, {cloud[:, 3].max():.2f}]")
    
    # 2. Render point cloud (interactive window if Open3D present, or headless verification)
    if HAS_OPEN3D:
        print("Rendering with Open3D Turbo colormap...")
        # To avoid blocking CI test runners, we can call it or pass headless flag
        render_intensity_pointcloud(cloud, sample_ratio=0.2, colormap_name="turbo")
    else:
        print("Open3D is not installed on this system.")
        print("Executing headless intensity pseudo-coloring verification:")
        intensities = cloud[:, 3]
        norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
        print(f"Normalized {len(norm)} points successfully for pseudo-coloring.")
        
    print("Chapter 2.4 verification completed successfully.\n")


if __name__ == "__main__":
    main()
