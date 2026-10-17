"""LidarLens Point Cloud Motion Deskewing (Chapters 5 & 9).

Performs point-level motion compensation via high-frequency IMU / odometry
SLERP trajectory interpolation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

logger = logging.getLogger("lidarlens.registration.deskew")


def de_skew_pointcloud(
    points: np.ndarray,
    timestamps: np.ndarray,
    pose_data: Dict[str, Any],
    t_ref: float,
) -> np.ndarray:
    """SLERP-interpolated motion deskewing using high-frequency IMU/odometry poses.

    Corrects distortion caused by ego-vehicle motion during LiDAR spinning/exposure.

    Args:
        points: (N, 3) or (N, 4) raw point cloud coordinates.
        timestamps: (N,) sampling timestamp per point.
        pose_data: Dictionary containing:
            - 't': (M,) timestamps array.
            - 'quat': (M, 4) quaternions (x, y, z, w).
            - 'pos': (M, 3) translation positions (x, y, z).
        t_ref: Reference target time (typically frame start or frame end).

    Returns:
        corrected_points: (N, 3) or (N, 4) motion-compensated coordinates.
    """
    xyz = points[:, :3].astype(np.float64)
    n = xyz.shape[0]

    t_raw = pose_data.get("t", pose_data.get("timestamps"))
    if t_raw is None:
        raise KeyError("pose_data must contain 't' or 'timestamps'")
    t_samples = np.asarray(t_raw, dtype=np.float64)

    rots_raw = pose_data.get("quat", pose_data.get("rotations"))
    if rots_raw is None:
        raise KeyError("pose_data must contain 'quat' or 'rotations'")
    if isinstance(rots_raw, R):
        rotations = rots_raw
    else:
        quats = np.asarray(rots_raw, dtype=np.float64)
        rotations = R.from_quat(quats)

    pos_raw = pose_data.get("pos", pose_data.get("translations"))
    if pos_raw is None:
        raise KeyError("pose_data must contain 'pos' or 'translations'")
    positions = np.asarray(pos_raw, dtype=np.float64)

    # 1. Initialize SLERP interpolator
    slerp = Slerp(t_samples, rotations)

    # 2. Reference global pose T(t_ref)
    R_ref = slerp(t_ref).as_matrix()
    p_ref = np.array([np.interp(t_ref, t_samples, positions[:, i]) for i in range(3)])

    # Clip point timestamps to valid trajectory interval
    ts_clipped = np.clip(timestamps, t_samples[0], t_samples[-1])

    # 3. Per-point sampled pose T(t_i)
    R_i = slerp(ts_clipped).as_matrix()  # (N, 3, 3)
    p_i = np.array([np.interp(ts_clipped, t_samples, positions[:, j]) for j in range(3)]).T  # (N, 3)

    # 4. Rigid transform: P_corrected = R_ref^T * (R_i * P_i + p_i - p_ref)
    points_in_world = np.einsum("nij,nj->ni", R_i, xyz) + p_i
    corrected_xyz = (points_in_world - p_ref) @ R_ref

    if points.shape[1] > 3:
        corrected_pts = points.copy()
        corrected_pts[:, :3] = corrected_xyz.astype(np.float32)
        return corrected_pts
    return corrected_xyz.astype(np.float32)
