"""
Chapter 8: 3D Object Detection Training & Deep Learning
Code Example 8.1: Minimum Viable Data Quality Sanity Check (KITTI Subset Inspection).

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第8章 8.2节 (Para 128)
"""

import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any


def check_kitti_subset(data_path: str) -> Dict[str, Any]:
    """Minimum viable data quality check (Sanity Check) for KITTI 3D dataset.
    
    Verifies point cloud counts, car instances, and distance distributions.
    
    Args:
        data_path: Root folder containing 'velodyne/' and 'label_2/' subdirectories.
        
    Returns:
        Dictionary of computed statistics.
    """
    lidar_files = sorted(list(Path(data_path).glob("velodyne/*.bin")))
    label_files = sorted(list(Path(data_path).glob("label_2/*.txt")))
    
    if len(lidar_files) == 0:
        raise FileNotFoundError(f"No .bin point cloud files found under {data_path}/velodyne/")
        
    stats: Dict[str, list] = {"pts_count": [], "car_instances": [], "distances": []}
    
    # Audit up to first 100 sample frames
    audit_files = list(zip(lidar_files[:100], label_files[:100])) if label_files else [(f, None) for f in lidar_files[:100]]
    
    for bin_p, label_p in audit_files:
        points = np.fromfile(bin_p, dtype=np.float32).reshape(-1, 4)
        stats["pts_count"].append(len(points))
        
        if label_p and Path(label_p).is_file():
            with open(label_p, "r", encoding="utf-8") as f:
                cars = [line for line in f.readlines() if line.startswith("Car")]
            stats["car_instances"].append(len(cars))
            for car in cars:
                # In KITTI label format: parts[11:14] are 3D camera coordinates (x, y, z)
                parts = car.split()
                if len(parts) >= 14:
                    loc = np.array(parts[11:14], dtype=np.float32)
                    stats["distances"].append(float(np.linalg.norm(loc)))
        else:
            stats["car_instances"].append(0)
            
    avg_pts = float(np.mean(stats["pts_count"]))
    total_cars = int(sum(stats["car_instances"]))
    med_dist = float(np.median(stats["distances"])) if stats["distances"] else 0.0
    
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"[Data Check] Evaluated Frames: {len(audit_files)}")
    print(f"[Data Check] Average Points per Frame: {avg_pts:.1f}")
    print(f"[Data Check] Total Car Instances: {total_cars}")
    print(f"[Data Check] Median Vehicle Distance: {med_dist:.2f} m")
    
    return {
        "frames_evaluated": len(audit_files),
        "avg_points": avg_pts,
        "total_cars": total_cars,
        "median_distance": med_dist,
        "stats": stats
    }


def create_synthetic_kitti_subset(temp_dir: str, num_frames: int = 5):
    """Creates a synthetic KITTI format directory structure for testing."""
    velo_dir = Path(temp_dir) / "velodyne"
    lbl_dir = Path(temp_dir) / "label_2"
    velo_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    for i in range(num_frames):
        # Generate random LiDAR scan (15,000 points)
        n_pts = int(rng.integers(12000, 18000))
        pts = np.zeros((n_pts, 4), dtype=np.float32)
        pts[:, 0] = rng.uniform(0, 50, n_pts)
        pts[:, 1] = rng.uniform(-20, 20, n_pts)
        pts[:, 2] = rng.uniform(-2, 3, n_pts)
        pts[:, 3] = rng.uniform(0, 1, n_pts)
        
        bin_path = velo_dir / f"{i:06d}.bin"
        pts.tofile(str(bin_path))
        
        # Generate KITTI label lines for Car
        # Format: type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
        lbl_path = lbl_dir / f"{i:06d}.txt"
        lines = []
        num_cars = int(rng.integers(1, 4))
        for _ in range(num_cars):
            x = rng.uniform(-10, 10)
            y = rng.uniform(-1, 2)
            z = rng.uniform(8, 35) # camera z is forward distance in KITTI
            lines.append(f"Car 0.00 0 0.00 50.0 100.0 150.0 200.0 1.6 1.8 4.2 {x:.2f} {y:.2f} {z:.2f} 0.1\n")
        
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def main():
    print("=== Chapter 8: KITTI Dataset Sanity Check ===")
    
    # Check if a real KITTI dataset directory exists; if not, use isolated synthetic sandbox
    real_kitti_path = "data/kitti/training"
    using_synthetic = False
    
    if os.path.exists(real_kitti_path) and len(list(Path(real_kitti_path).glob("velodyne/*.bin"))) > 0:
        target_path = real_kitti_path
    else:
        temp_dir = tempfile.mkdtemp(prefix="kitti_sanity_test_")
        create_synthetic_kitti_subset(temp_dir, num_frames=10)
        target_path = temp_dir
        using_synthetic = True
        
    try:
        report = check_kitti_subset(target_path)
        
        # Statistical assertions
        assert report["frames_evaluated"] > 0
        assert report["avg_points"] > 1000.0, "Point cloud count suspiciously low!"
        assert report["total_cars"] >= 1, "At least 1 car instance should be detected in the subset."
        assert 5.0 <= report["median_distance"] <= 60.0, "Vehicle distance median outside normal perception range."
        
        print("Data quality sanity check passed all criteria.")
        print("Chapter 8.1 verification completed successfully.\n")
    finally:
        if using_synthetic and os.path.exists(target_path):
            shutil.rmtree(target_path)


if __name__ == "__main__":
    main()
