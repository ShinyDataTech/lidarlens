"""LidarLens — Production-Grade 3D Point Cloud Perception, Odometry, Mapping, and Agentic AI.

Curriculum-aligned with "Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战".

Quick Start:
    import lidarlens as ll

    # Zero-copy memory buffer & streaming
    buffer = ll.LidarBuffer(capacity=100_000)
    streamer = ll.Streamer("data/kitti/velodyne", format="kitti")

    # Sliding-window temporal pipeline (>=30 Hz)
    pipeline = ll.TemporalPipeline(window_size=5)
    pipeline.add(ll.filters.VoxelDownsample(voxel_size=0.1))
    pipeline.add(ll.filters.DynamicTemporalSOR(base_neighbors=30, base_std_ratio=2.0))

    for raw_frame in streamer:
        cleaned = pipeline.process(raw_frame)
"""

from lidarlens._version import __version__

# ── Core Data Structures (Chapters 2 & 3) ───────────────────────────────────
from lidarlens.core import LidarBuffer, LidarFrame
from lidarlens.pipeline import TemporalPipeline, RingBuffer
from lidarlens.benchmark import LidarLensBenchmark

# ── Submodules ──────────────────────────────────────────────────────────────
from lidarlens import filters
from lidarlens import terrain
from lidarlens import geometry
from lidarlens import cluster
from lidarlens import registration
from lidarlens import mapping
from lidarlens import augment
from lidarlens import models
from lidarlens import sim2real
from lidarlens import scenegraph
from lidarlens import agent
from lidarlens import deploy
from lidarlens import ros2
from lidarlens import vis
from lidarlens import dataset
from lidarlens import io

# Top-level direct streaming & pipeline access
from lidarlens.io.streamer import Streamer
from lidarlens.filters.downsample import VoxelDownsample
from lidarlens.filters.sor import DynamicTemporalSOR, StatisticalOutlierRemoval
from lidarlens.registration.gicp import LowLatencyGICP

# ── Legacy Geospatial Functions (Backward Compatibility) ─────────────────────
from lidarlens.io.las_io import read, save, read_header, read_bounds
from lidarlens.preprocessing import (
    voxel_downsample,
    uniform_downsample,
    random_downsample,
    denoise,
    remove_statistical_outliers,
    remove_radius_outliers,
)
downsample = voxel_downsample

from lidarlens.segmentation import (
    segment_plane,
    extract_ground,
    cluster_dbscan,
    extract_clusters,
    segment_heuristic,
)
from lidarlens.analysis import (
    compute_statistics,
    elevation_histogram,
    compare_clouds,
    point_density,
    roughness,
    cross_section,
    compute_volume,
)
stats = compute_statistics

from lidarlens.classification import (
    ASPRS_CLASSES,
    reclassify_in_polygon,
    reclassify_by_elevation,
    get_class_distribution,
    merge_classes,
)

__all__ = [
    "__version__",
    # Core
    "LidarBuffer",
    "LidarFrame",
    "TemporalPipeline",
    "RingBuffer",
    "Streamer",
    "LidarLensBenchmark",
    # Submodules
    "filters",
    "terrain",
    "geometry",
    "cluster",
    "registration",
    "mapping",
    "augment",
    "models",
    "sim2real",
    "scenegraph",
    "agent",
    "deploy",
    "ros2",
    "vis",
    "dataset",
    "io",
    # Legacy
    "read",
    "save",
    "read_header",
    "read_bounds",
    "voxel_downsample",
    "uniform_downsample",
    "random_downsample",
    "denoise",
    "downsample",
    "remove_statistical_outliers",
    "remove_radius_outliers",
    "segment_plane",
    "extract_ground",
    "cluster_dbscan",
    "extract_clusters",
    "segment_heuristic",
    "compute_statistics",
    "stats",
    "elevation_histogram",
    "compare_clouds",
    "point_density",
    "roughness",
    "cross_section",
    "compute_volume",
    "ASPRS_CLASSES",
    "reclassify_in_polygon",
    "reclassify_by_elevation",
    "get_class_distribution",
    "merge_classes",
]
