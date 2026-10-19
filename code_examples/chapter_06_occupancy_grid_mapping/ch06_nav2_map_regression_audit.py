"""
Chapter 6: Occupancy Grid Mapping & Quality Audit
Code Example 6.3: Automated Nav2 PGM/YAML Map Regression & Quality Audit.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第6章 6.3.2节
"""

import os
import tempfile
import numpy as np
import yaml
import cv2
import lidarlens as ll
from lidarlens.io.map_io import save_nav2_map, run_map_regression


def create_sample_nav2_map(output_dir: str):
    """Generate a realistic Nav2 OccupancyGrid map (PGM + YAML) for offline audit."""
    h, w = 400, 400
    # Standard Nav2 grayscale values:
    # 254 (0.0 cost / free)
    # 0 (1.0 cost / lethal)
    # 205 (unknown / -1)
    img = np.full((h, w), 205, dtype=np.uint8) # Default unknown
    
    # Large connected free room: [50:350, 50:350]
    img[50:350, 50:350] = 254
    
    # Boundary walls (lethal obstacles = 0)
    img[50:52, 50:350] = 0
    img[348:350, 50:350] = 0
    img[50:350, 50:52] = 0
    img[50:350, 348:350] = 0
    
    # Interior pillars
    img[120:135, 120:135] = 0
    img[260:275, 260:275] = 0
    
    # Inject isolated single-pixel salt-and-pepper noise islands (should be flagged by regression)
    for _ in range(8):
        ry = np.random.randint(70, 330)
        rx = np.random.randint(70, 330)
        img[ry:ry+2, rx:rx+2] = 0
        
    pgm_path = os.path.join(output_dir, "test_map.pgm")
    yaml_path = os.path.join(output_dir, "test_map.yaml")
    
    cv2.imwrite(pgm_path, img)
    
    meta = {
        "image": "test_map.pgm",
        "resolution": 0.05,
        "origin": [-10.0, -10.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "mode": "trinary"
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)
        
    return pgm_path, meta


def main():
    print("=== Chapter 6: Automated Nav2 Map Regression Audit ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pgm_path, yaml_cfg = create_sample_nav2_map(tmpdir)
        print(f"Synthesized Nav2 map artifact at: {pgm_path}")
        
        # Run Automated Map Quality Audit
        audit = run_map_regression(pgm_path=pgm_path, yaml_cfg=yaml_cfg)
        
        print("\n" + "=" * 50)
        print("          Nav2 Offline Map Regression Report")
        print("=" * 50)
        print(f"Free Space Ratio:       {audit['free_ratio']*100:.1f}%")
        print(f"Unknown Area Ratio:     {audit['unknown_ratio']*100:.1f}%")
        print(f"Lethal Obstacle Ratio:  {audit['lethal_ratio']*100:.1f}%")
        print(f"Total Connected Bodies: {audit['connected_components']}")
        print(f"Isolated Noise Islands: {audit['isolated_noise_islands']} (Pruned threshold < 5 cells)")
        print(f"Audit Quality Verdict:  {'PASS' if audit['audit_passed'] else 'FAIL'}")
        print("=" * 50)
        
        assert audit["free_ratio"] > 0.40, "Free space ratio unexpectedly low!"
        assert audit["connected_components"] > 1, "Failed to identify room connected components!"

    print("Chapter 6.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
