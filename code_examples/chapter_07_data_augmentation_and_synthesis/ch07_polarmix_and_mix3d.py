"""
Chapter 7: 3D Data Augmentation & Synthesis Quality Gates
Code Example 7.2: PolarMix Sector Swapping & Mix3D Scene Superposition.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第7章 7.2.2 & 7.2.3节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.augment import polar_mix, mix3d


def make_scene(offset_x: float = 0.0, num_pts: int = 8000) -> dict:
    """Generate sample scene points and bounding boxes."""
    pts = np.zeros((num_pts, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(5, 35, num_pts) + offset_x
    pts[:, 1] = np.random.uniform(-15, 15, num_pts)
    pts[:, 2] = np.random.uniform(-1.5, 2.5, num_pts)
    pts[:, 3] = np.random.uniform(0.1, 0.9, num_pts)
    
    boxes = np.array([
        [15.0 + offset_x, -2.0, 0.0, 4.2, 1.8, 1.5, 0.1],
        [25.0 + offset_x,  3.0, 0.2, 1.9, 0.8, 1.4, -0.2]
    ], dtype=np.float32)
    names = ["Car", "Cyclist"]
    return {"points": pts, "gt_boxes": boxes, "gt_names": names}


def main():
    print("=== Chapter 7: PolarMix & Mix3D Data Augmentation ===")
    
    scene_a = make_scene(offset_x=0.0, num_pts=6000)
    scene_b = make_scene(offset_x=5.0, num_pts=6000)
    
    # 1. PolarMix Sector Cut & Paste
    # Slices an azimuth sector [0, pi/2] from Scene B into Scene A
    t0 = time.perf_counter()
    pm_res = polar_mix(
        data_dict1=scene_a,
        data_dict2=scene_b,
        sector_angles=(0.0, np.pi / 2)
    )
    pm_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"PolarMix completed in {pm_ms:.2f} ms:")
    print(f"  Fused Points: {len(pm_res['points']):,} (Scene A: {len(scene_a['points'])}, Scene B: {len(scene_b['points'])})")
    print(f"  Fused Boxes:  {len(pm_res['gt_boxes'])}")
    assert len(pm_res["points"]) > 0
    assert len(pm_res["gt_boxes"]) > 0
    
    # 2. Mix3D Scene Superposition
    # Overlays two entire 3D scenes with union of bounding boxes
    t0 = time.perf_counter()
    m3d_res = mix3d(
        data_dict1=scene_a,
        data_dict2=scene_b
    )
    m3d_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\nMix3D completed in {m3d_ms:.2f} ms:")
    print(f"  Total Superposed Points: {len(m3d_res['points']):,}")
    print(f"  Total Superposed Boxes:  {len(m3d_res['gt_boxes'])}")
    
    assert len(m3d_res["points"]) == len(scene_a["points"]) + len(scene_b["points"])
    assert len(m3d_res["gt_boxes"]) == len(scene_a["gt_boxes"]) + len(scene_b["gt_boxes"])
    print("Chapter 7.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
