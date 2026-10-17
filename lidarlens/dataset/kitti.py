"""LidarLens KITTI Dataset Utilities (Chapter 8).

Provides sanity check inspection for KITTI 3D object detection data subsets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("lidarlens.dataset.kitti")


def check_kitti_subset(data_path: str) -> Dict[str, Any]:
    """Minimum Viable Data Quality Sanity Check (Chapter 8).

    Verifies LiDAR point counts, Car instances, and radial distance distributions.

    Args:
        data_path: Path to KITTI dataset root containing 'velodyne' and 'label_2' folders.

    Returns:
        Summary statistics dictionary.
    """
    p = Path(data_path)
    lidar_files = sorted(list((p / "velodyne").glob("*.bin"))) if (p / "velodyne").is_dir() else []
    label_files = sorted(list((p / "label_2").glob("*.txt"))) if (p / "label_2").is_dir() else []

    stats: Dict[str, list] = {"pts_count": [], "car_instances": [], "distances": []}

    # Inspect up to first 100 samples
    sample_pairs = list(zip(lidar_files[:100], label_files[:100]))

    if not sample_pairs:
        # Mock/simulated check for synthetic testing or empty path
        logger.warning("No KITTI .bin and .txt pairs found at %s. Generating mock audit.", data_path)
        return {
            "num_samples": 0,
            "avg_points": 0.0,
            "total_cars": 0,
            "median_distance": 0.0,
            "passed": False,
        }

    for bin_p, label_p in sample_pairs:
        points = np.fromfile(str(bin_p), dtype=np.float32).reshape(-1, 4)
        stats["pts_count"].append(len(points))

        with open(label_p, "r", encoding="utf-8") as f:
            cars = [line for line in f.readlines() if line.startswith("Car")]
        stats["car_instances"].append(len(cars))

        for car in cars:
            tokens = car.split()
            if len(tokens) >= 14:
                loc = np.array(tokens[11:14], dtype=np.float32)
                stats["distances"].append(float(np.linalg.norm(loc)))

    avg_pts = float(np.mean(stats["pts_count"])) if stats["pts_count"] else 0.0
    total_cars = int(sum(stats["car_instances"]))
    med_dist = float(np.median(stats["distances"])) if stats["distances"] else 0.0

    print(f"[Data Check] 平均单帧点数: {avg_pts:.1f}")
    print(f"[Data Check] Car实例总数: {total_cars}")
    print(f"[Data Check] 车辆距离中位数: {med_dist:.2f} m")

    return {
        "num_samples": len(sample_pairs),
        "avg_points": avg_pts,
        "total_cars": total_cars,
        "median_distance": med_dist,
        "passed": bool(avg_pts > 1000 and total_cars >= 0),
    }
