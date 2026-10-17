"""LidarLens Map I/O & Regression Testing (Chapter 6).

Provides automated PGM/YAML offline map regression audit, connected component
analysis, island noise detection, and Nav2 standard map serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("lidarlens.io.map_io")

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None


def run_map_regression(pgm_path: str, yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Automated PGM/YAML offline map regression audit (Chapter 6).

    Args:
        pgm_path: Path to the 8-bit PGM grayscale occupancy grid map.
        yaml_cfg: Map configuration dictionary containing 'occupied_thresh',
            'free_thresh', 'resolution', etc.

    Returns:
        Audit results dictionary containing:
            - 'total_components': Count of connected obstacle clusters.
            - 'island_noise_count': Count of isolated micro-noise clusters (area < 5).
            - 'unknown_coverage': Ratio of unexplored/unknown map cells (value 205).
            - 'passed': Boolean pass/fail gate.
    """
    path = Path(pgm_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot load PGM map from {pgm_path}")

    # Read image using OpenCV or manual binary parse
    if _HAS_CV2 and cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        # Fallback simple PGM P5 binary parser
        with open(path, "rb") as f:
            header = f.readline().decode().strip()
            if header not in ("P5", "P2"):
                raise ValueError(f"Unsupported PGM header: {header}")
            line = f.readline().decode()
            while line.startswith("#"):
                line = f.readline().decode()
            w, h = map(int, line.split())
            max_val = int(f.readline().decode().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = data.reshape((h, w))

    if img is None:
        raise FileNotFoundError(f"Failed decoding PGM map from {pgm_path}")

    # 2. Extract obstacle mask according to YAML threshold
    # Standard Nav2 convention: in PGM, 0=occupied (dark), 254=free (white), 205=unknown (grey)
    occ_thresh = yaml_cfg.get("occupied_thresh", 0.65)
    binary_map = (img < int(255 * (1.0 - occ_thresh))).astype(np.uint8)

    # 3. Connected components analysis for island noise (Island Noise)
    if _HAS_CV2 and cv2 is not None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
        small_islands = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                small_islands += 1
        total_components = num_labels
    else:
        from scipy.ndimage import label
        labeled_arr, num_features = label(binary_map)
        unique, counts = np.unique(labeled_arr, return_counts=True)
        small_islands = int(np.sum((counts[1:] < 5)))
        total_components = num_features + 1

    # 4. Unknown area coverage
    unknown_rate = float(np.mean(img == 205))

    free_rate = float(np.mean(img == 254))
    lethal_rate = float(np.mean(img == 0))
    passed = bool(small_islands < 10 and unknown_rate < 0.8)

    audit_results = {
        "total_components": int(total_components),
        "connected_components": int(total_components),
        "island_noise_count": int(small_islands),
        "isolated_noise_islands": int(small_islands),
        "unknown_coverage": float(unknown_rate),
        "unknown_ratio": float(unknown_rate),
        "free_ratio": float(free_rate),
        "lethal_ratio": float(lethal_rate),
        "passed": passed,
        "audit_passed": passed,
    }

    logger.info(
        "[Map Audit] Total components: %d, Island noise clusters: %d, Unknown coverage: %.2f%%, Passed: %s",
        total_components,
        small_islands,
        unknown_rate * 100.0,
        passed,
    )

    return audit_results


def save_nav2_map(
    grid: np.ndarray,
    meta: Dict[str, Any],
    pgm_path: str,
    yaml_path: str,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
) -> None:
    """Save 2D OccupancyGrid matrix as standard ROS 2 / Nav2 PGM + YAML map pair.

    Args:
        grid: (H, W) array where 0=free, 100=lethal, -1=unknown.
        meta: Dict containing 'resolution', 'origin', etc.
        pgm_path: Destination .pgm file.
        yaml_path: Destination .yaml file.
        occupied_thresh: Probability threshold for occupied cells.
        free_thresh: Probability threshold for free cells.
    """
    h, w = grid.shape
    pgm_img = np.full((h, w), 205, dtype=np.uint8)  # Default unknown (205)

    pgm_img[grid == 0] = 254    # Free (white)
    pgm_img[grid >= 50] = 0     # Occupied/lethal (black)

    # Invert rows to match ROS 2 bottom-left origin convention
    pgm_flipped = np.flipud(pgm_img)

    if _HAS_CV2 and cv2 is not None:
        cv2.imwrite(str(pgm_path), pgm_flipped)
    else:
        # Simple binary P5 PGM writer
        with open(pgm_path, "wb") as f:
            header = f"P5\n{w} {h}\n255\n".encode("ascii")
            f.write(header)
            f.write(pgm_flipped.tobytes())

    # Build YAML manifest
    res = float(meta.get("resolution", 0.05))
    origin = meta.get("origin", (0.0, 0.0))
    origin_list = [float(origin[0]), float(origin[1]), 0.0]

    yaml_data = {
        "image": Path(pgm_path).name,
        "mode": "trinary",
        "resolution": res,
        "origin": origin_list,
        "negate": 0,
        "occupied_thresh": float(occupied_thresh),
        "free_thresh": float(free_thresh),
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    logger.info("Saved Nav2 standard map pair: %s and %s", pgm_path, yaml_path)


def load_nav2_map(yaml_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load ROS 2 Nav2 standard map from YAML descriptor and associated PGM image."""
    ypath = Path(yaml_path)
    with open(ypath, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img_name = cfg.get("image", "")
    pgm_path = ypath.parent / img_name
    if not pgm_path.is_file():
        raise FileNotFoundError(f"Associated PGM image {pgm_path} does not exist.")

    if _HAS_CV2 and cv2 is not None:
        img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    else:
        with open(pgm_path, "rb") as f:
            header = f.readline().decode().strip()
            line = f.readline().decode()
            while line.startswith("#"):
                line = f.readline().decode()
            w, h = map(int, line.split())
            max_val = int(f.readline().decode().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = data.reshape((h, w))

    # Un-flip rows to restore standard indexing
    img_restored = np.flipud(img)

    occ_thresh = cfg.get("occupied_thresh", 0.65)
    free_thresh = cfg.get("free_thresh", 0.196)

    grid = np.full(img_restored.shape, -1, dtype=np.int8)
    grid[img_restored >= int(255 * (1.0 - free_thresh))] = 0
    grid[img_restored <= int(255 * (1.0 - occ_thresh))] = 100

    return grid, cfg
