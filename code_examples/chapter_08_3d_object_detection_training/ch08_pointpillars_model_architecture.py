"""
Chapter 8: 3D Object Detection Training & Deep Learning
Code Example 8.2: PointPillars End-to-End Perception Architecture Pipeline.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第8章 8.1 & 8.3节
"""

import time
import numpy as np
import torch
import lidarlens as ll
from lidarlens.models import (
    pillarize_points,
    PillarFeatureNet,
    ScatterBEV,
    PointPillarsBackbone
)


def generate_synthetic_scan(num_points: int = 25_000) -> np.ndarray:
    """Generate synthetic LiDAR point cloud within PointPillars perception range."""
    rng = np.random.default_rng(42)
    pts = np.zeros((num_points, 4), dtype=np.float32)
    # Range: x in [0, 69.12], y in [-39.68, 39.68], z in [-3.0, 1.0]
    pts[:, 0] = rng.uniform(0.5, 68.0, num_points)
    pts[:, 1] = rng.uniform(-38.0, 38.0, num_points)
    pts[:, 2] = rng.uniform(-2.5, 0.5, num_points)
    pts[:, 3] = rng.uniform(0.1, 0.9, num_points) # intensity
    return pts


def main():
    print("=== Chapter 8: PointPillars Perception Architecture ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")
    
    # Grid specification matching KITTI / autonomous driving standards
    voxel_size = (0.16, 0.16, 4.0)
    pc_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
    grid_w = int(np.ceil((pc_range[3] - pc_range[0]) / voxel_size[0])) # 432
    grid_h = int(np.ceil((pc_range[4] - pc_range[1]) / voxel_size[1])) # 496
    print(f"BEV Canvas Resolution: {grid_h} x {grid_w} (dx=0.16m, dy=0.16m)")
    
    # 1. Generate Input Data
    points = generate_synthetic_scan(25_000)
    print(f"Generated Raw Scan: {len(points):,} points")
    
    # 2. Stage 1: Dynamic Pillarization & 9D Feature Augmentation
    t0 = time.perf_counter()
    pillar_feats_np, pillar_coords_np, pillar_counts_np = pillarize_points(
        points,
        voxel_size=voxel_size,
        point_cloud_range=pc_range,
        max_points_per_pillar=32,
        max_pillars=16000
    )
    pillar_ms = (time.perf_counter() - t0) * 1000.0
    
    # Number of valid, populated pillars
    active_pillars = int(np.sum(pillar_counts_np > 0))
    print(f"\nStage 1: Pillarization completed in {pillar_ms:.2f} ms")
    print(f"  Active Non-Empty Pillars: {active_pillars:,} / 16,000 max capacity")
    print(f"  Pillar Feature Tensor Shape: {pillar_feats_np.shape} (9D per point)")
    
    # Convert active pillars to torch tensors
    active_mask = pillar_counts_np > 0
    p_feats = torch.from_numpy(pillar_feats_np[active_mask]).to(device)
    p_coords = torch.from_numpy(pillar_coords_np[active_mask]).to(device)
    
    # 3. Stage 2: Pillar Feature Network (PFN: 9D -> 64D pooled)
    pfn = PillarFeatureNet(in_channels=9, feat_channels=64).to(device)
    pfn.eval()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        pooled_feats = pfn(p_feats)
    pfn_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\nStage 2: PFN Linear & Max-Pool completed in {pfn_ms:.2f} ms")
    print(f"  Pooled Features Shape: {tuple(pooled_feats.shape)}")
    assert pooled_feats.shape == (active_pillars, 64)
    
    # 4. Stage 3: ScatterBEV (1D -> 2D Canvas)
    scatter = ScatterBEV(grid_h=grid_h, grid_w=grid_w, feat_channels=64).to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        pseudo_image = scatter(pooled_feats, p_coords)
    scatter_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\nStage 3: Scatter to 2D Pseudo-Image completed in {scatter_ms:.2f} ms")
    print(f"  Pseudo-Image Canvas Shape: {tuple(pseudo_image.shape)}")
    assert pseudo_image.shape == (1, 64, grid_h, grid_w)
    
    # 5. Stage 4: 2D Convolutional Backbone (64 -> 256 channels)
    backbone = PointPillarsBackbone(in_channels=64, out_channels=256).to(device)
    backbone.eval()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        bev_features = backbone(pseudo_image)
    backbone_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\nStage 4: PointPillars Backbone Conv completed in {backbone_ms:.2f} ms")
    print(f"  High-level BEV Feature Map Shape: {tuple(bev_features.shape)}")
    assert bev_features.shape == (1, 256, grid_h, grid_w)
    
    total_pipeline_ms = pillar_ms + pfn_ms + scatter_ms + backbone_ms
    print(f"\nTotal PointPillars End-to-End Pipeline Latency: {total_pipeline_ms:.2f} ms")
    print("Chapter 8.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
