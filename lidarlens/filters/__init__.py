"""LidarLens Filters Subsystem.

Combines spatial downsampling, statistical outlier removal (dynamic/temporal),
rain/fog cascading denoisers, and ground plane segmentation.
"""

from lidarlens.filters.downsample import VoxelDownsample
from lidarlens.filters.sor import (
    DynamicTemporalSOR,
    StatisticalOutlierRemoval,
    rain_pointcloud_denoise,
)
from lidarlens.filters.ground import RANSACGround, PatchworkPlusPlus

# Re-export legacy functions for backward compatibility
from lidarlens.filtering import (
    crop_by_bounds,
    filter_by_elevation,
    filter_by_classification,
    filter_by_return_number,
    filter_by_intensity,
    crop_by_polygon,
)

__all__ = [
    "VoxelDownsample",
    "DynamicTemporalSOR",
    "StatisticalOutlierRemoval",
    "rain_pointcloud_denoise",
    "RANSACGround",
    "PatchworkPlusPlus",
    "crop_by_bounds",
    "filter_by_elevation",
    "filter_by_classification",
    "filter_by_return_number",
    "filter_by_intensity",
    "crop_by_polygon",
]
