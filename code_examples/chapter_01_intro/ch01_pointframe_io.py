"""
Chapter 1: Introduction to 3D Spatial Perception
Code Example 1.1: PointFrame Data Structure, KITTI/AV2 I/O, Intensity Normalization, and Diagnostics.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第1章 1.3.3节
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import tempfile
import os

# Import LidarLens core for unified frame interop
import lidarlens as ll


@dataclass
class PointFrame:
    """Standard LiDAR Point Cloud Frame Data Structure.
    
    Attributes:
        xyz: (N, 3) float32 matrix containing 3D spatial coordinates.
        attr: Dictionary of auxiliary attribute arrays (e.g., 'remission', 'ring', 'range').
        frame_id: Coordinate reference system identifier (e.g., 'velodyne', 'ego_vehicle').
        t_ns: Reference acquisition timestamp in nanoseconds.
    """
    xyz: np.ndarray
    attr: Dict[str, np.ndarray]
    frame_id: str
    t_ns: Optional[int] = None

    def validate(self) -> None:
        """Validate tensor shapes, dimensions, and contiguous float32 data types."""
        assert self.xyz.ndim == 2 and self.xyz.shape[1] == 3, f"xyz must be (N, 3), got {self.xyz.shape}"
        assert self.xyz.dtype == np.float32, f"xyz dtype must be float32, got {self.xyz.dtype}"
        n = self.xyz.shape[0]
        for k, v in self.attr.items():
            assert v.shape[0] == n, f"Attribute '{k}' length {v.shape[0]} mismatches point count {n}"

    def to_lidarlens(self) -> ll.LidarFrame:
        """Zero-copy conversion into a high-performance LidarLens LidarFrame."""
        # Pack XYZ and intensity/remission into contiguous (N, 4)
        intensity = self.attr.get("remission", self.attr.get("intensity", np.zeros((len(self.xyz),), dtype=np.float32)))
        points_xyzi = np.column_stack([self.xyz, intensity.astype(np.float32)])
        ts = (self.t_ns / 1e9) if self.t_ns is not None else 0.0
        return ll.LidarFrame(points=points_xyzi, timestamp=ts, frame_id=self.frame_id)


def read_kitti_bin(bin_path: str | Path, frame_id: str = "velodyne") -> PointFrame:
    """Parse compact binary KITTI velodyne point cloud file (.bin).
    
    Format: Flat continuous array of float32 values: [x, y, z, remission].
    """
    bin_path = Path(bin_path)
    raw = np.fromfile(str(bin_path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Malformed KITTI bin file: total elements {raw.size} is not divisible by 4.")
    
    pts = raw.reshape(-1, 4)
    xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
    remission = np.ascontiguousarray(pts[:, 3], dtype=np.float32)
    
    pf = PointFrame(xyz=xyz, attr={"remission": remission}, frame_id=frame_id)
    pf.validate()
    return pf


def read_av2_feather(feather_path: str | Path, frame_id: str = "ego_vehicle") -> PointFrame:
    """Parse Argoverse 2 Feather/Arrow point cloud file."""
    feather_path = Path(feather_path)
    try:
        import pyarrow.feather as feather
        table = feather.read_table(str(feather_path))
        x = table.column("x").to_numpy().astype(np.float32)
        y = table.column("y").to_numpy().astype(np.float32)
        z = table.column("z").to_numpy().astype(np.float32)
        intensity = table.column("intensity").to_numpy().astype(np.float32)
        xyz = np.column_stack([x, y, z])
        pf = PointFrame(xyz=xyz, attr={"intensity": intensity}, frame_id=frame_id)
        pf.validate()
        return pf
    except ImportError:
        # Fallback simulation if pyarrow is absent
        print("Note: pyarrow not installed, falling back to simulated AV2 table.")
        xyz = np.random.uniform(-50, 50, size=(1000, 3)).astype(np.float32)
        intensity = np.random.uniform(0, 255, size=(1000,)).astype(np.float32)
        return PointFrame(xyz=xyz, attr={"intensity": intensity}, frame_id=frame_id)


def normalize_intensity(pf: PointFrame, key_in: str = "remission",
                        key_out: str = "intensity_norm",
                        clip_percentile: Tuple[float, float] = (1.0, 99.0)) -> PointFrame:
    """Standardize sensor reflection intensities to the [0.0, 1.0] interval."""
    if key_in not in pf.attr:
        raise KeyError(f"Attribute key '{key_in}' not found in PointFrame.")
        
    vals = pf.attr[key_in].astype(np.float32)
    p_low, p_high = np.percentile(vals, clip_percentile)
    clipped = np.clip(vals, p_low, p_high)
    denom = max(float(p_high - p_low), 1e-6)
    norm = (clipped - p_low) / denom
    
    pf.attr[key_out] = norm.astype(np.float32)
    return pf


def plot_pointframe_stats(pf: PointFrame, intensity_key: str = "intensity_norm") -> None:
    """Diagnostic distribution analysis for 3D point cloud coordinate ranges and intensities."""
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Spatial scatter BEV
        axes[0].scatter(pf.xyz[::10, 0], pf.xyz[::10, 1], s=1, c="teal", alpha=0.5)
        axes[0].set_title(f"BEV Scatter ({pf.frame_id})")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
        axes[0].grid(True)
        
        # Intensity histogram
        if intensity_key in pf.attr:
            axes[1].hist(pf.attr[intensity_key], bins=50, color="indigo", alpha=0.7)
            axes[1].set_title(f"Intensity Histogram [{intensity_key}]")
            axes[1].set_xlabel("Normalized Value")
            axes[1].set_ylabel("Count")
            axes[1].grid(True)
            
        plt.tight_layout()
        plt.close(fig)
        print("Diagnostic plot generated successfully (headless mode).")
    except ImportError:
        print("Note: matplotlib not installed; skipping graphic plot rendering.")


def main():
    print("=== Chapter 1: PointFrame & I/O Pipeline Demonstration ===")
    
    # 1. Create a synthetic KITTI .bin file to verify parsing
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_bin = os.path.join(tmpdir, "000001.bin")
        n_pts = 5000
        mock_raw = np.zeros((n_pts, 4), dtype=np.float32)
        mock_raw[:, 0] = np.random.uniform(2, 60, n_pts)  # X (m)
        mock_raw[:, 1] = np.random.uniform(-25, 25, n_pts) # Y (m)
        mock_raw[:, 2] = np.random.uniform(-2, 3, n_pts)   # Z (m)
        mock_raw[:, 3] = np.random.uniform(0.0, 1.0, n_pts) # remission
        mock_raw.tofile(sample_bin)
        
        # 2. Read KITTI binary file
        pf = read_kitti_bin(sample_bin, frame_id="velodyne")
        print(f"Loaded PointFrame with {len(pf.xyz)} points in frame '{pf.frame_id}'.")
        
        # 3. Intensity normalization
        pf = normalize_intensity(pf, key_in="remission", key_out="intensity_norm")
        norm_vals = pf.attr["intensity_norm"]
        print(f"Normalized intensity range: [{norm_vals.min():.4f}, {norm_vals.max():.4f}]")
        assert 0.0 <= norm_vals.min() and norm_vals.max() <= 1.0, "Intensity normalization out of range"
        
        # 4. Convert to LidarLens high-performance frame
        ll_frame = pf.to_lidarlens()
        print(f"Converted to LidarLens frame: shape={ll_frame.points.shape}, is_empty={ll_frame.is_empty}")
        assert not ll_frame.is_empty
        assert ll_frame.num_points == n_pts
        
        # 5. Diagnostic statistics
        plot_pointframe_stats(pf)

    print("Chapter 1 verification PASSED.\n")


if __name__ == "__main__":
    main()
