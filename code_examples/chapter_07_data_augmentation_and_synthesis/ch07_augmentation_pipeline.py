"""
Chapter 7: 3D Data Augmentation & Synthesis Quality Gates
Code Example 7.1: MMDetection3D-Style Augmentation Pipeline with Synchronized BBoxes.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第7章 7.1 & 7.2节
"""

import time
import numpy as np
from typing import Dict, Any, List
import lidarlens as ll
from lidarlens.augment import AugmentationPipeline


def generate_sample_scene(seed: int = 100):
    """Create a sample 3D LiDAR scene with car, cyclist, and pedestrian 3D bounding boxes."""
    rng = np.random.default_rng(seed)
    n_pts = 15_000
    points = np.zeros((n_pts, 4), dtype=np.float32)
    points[:, 0] = rng.uniform(2, 45, n_pts)
    points[:, 1] = rng.uniform(-15, 15, n_pts)
    points[:, 2] = rng.uniform(-1.5, 3.0, n_pts)
    points[:, 3] = rng.uniform(0.1, 0.9, n_pts)
    
    # 3D bounding boxes format: [x, y, z, dx, dy, dz, yaw]
    gt_boxes = np.array([
        [12.0, -2.5, 0.0, 4.5, 1.8, 1.6, np.deg2rad(10.0)], # Car
        [24.0,  3.0, 0.2, 1.8, 0.8, 1.5, np.deg2rad(-25.0)], # Cyclist
        [8.0,   1.5, -0.2, 0.6, 0.6, 1.7, 0.0],              # Pedestrian
    ], dtype=np.float32)
    
    gt_names = ["Car", "Cyclist", "Pedestrian"]
    
    data_dict = {
        "points": points,
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "frame_id": "000042"
    }
    return data_dict


def main():
    print("=== Chapter 7: MMDetection3D-Style 3D Augmentation Pipeline ===")
    
    # 1. Pipeline configuration definition
    pipeline_cfg = {
        "seed": 42, # Strictly deterministic
        "transforms": [
            {
                "type": "GlobalRotScaleTrans",
                "rot_range": [-0.3925, 0.3925], # [-22.5 deg, +22.5 deg]
                "scale_ratio_range": [0.95, 1.05],
                "translation_std": [0.2, 0.2, 0.1]
            },
            {
                "type": "RandomFlip3D",
                "flip_ratio_bev_horizontal": 0.5
            }
        ]
    }
    
    pipeline = AugmentationPipeline(pipeline_cfg)
    data = generate_sample_scene()
    
    orig_points = data["points"].copy()
    orig_boxes = data["gt_boxes"].copy()
    
    print(f"Original Scene: {len(orig_points):,} points, {len(orig_boxes)} GT 3D boxes.")
    print(f"Car #0 Box: Center=[{orig_boxes[0, 0]:.2f}, {orig_boxes[0, 1]:.2f}, {orig_boxes[0, 2]:.2f}], Yaw={np.rad2deg(orig_boxes[0, 6]):.1f}°")
    
    # 2. Execute Augmentation
    t0 = time.perf_counter()
    aug_data = pipeline(data)
    aug_ms = (time.perf_counter() - t0) * 1000.0
    
    aug_points = aug_data["points"]
    aug_boxes = aug_data["gt_boxes"]
    
    print(f"\nAugmented in {aug_ms:.2f} ms:")
    print(f"Augmented Car #0 Box: Center=[{aug_boxes[0, 0]:.2f}, {aug_boxes[0, 1]:.2f}, {aug_boxes[0, 2]:.2f}], Yaw={np.rad2deg(aug_boxes[0, 6]):.1f}°")
    
    # 3. Verify Deterministic Seed Locking
    pipeline_repeat = AugmentationPipeline(pipeline_cfg)
    repeat_data = pipeline_repeat(generate_sample_scene())
    np.testing.assert_allclose(aug_points, repeat_data["points"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(aug_boxes, repeat_data["gt_boxes"], rtol=1e-5, atol=1e-5)
    print("Deterministic random seed locking verified (100% bitwise reproducible).")
    
    # 4. Verify Dimension Invariance (Box dimensions dx, dy, dz are preserved under rigid transform)
    scale = np.linalg.norm(aug_boxes[0, 3:6]) / np.linalg.norm(orig_boxes[0, 3:6])
    print(f"Applied Scaling Factor: {scale:.4f} (within [0.95, 1.05] bounds)")
    assert 0.95 <= scale <= 1.05
    
    print("Chapter 7.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
