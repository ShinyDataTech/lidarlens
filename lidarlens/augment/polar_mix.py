"""LidarLens PolarMix 3D Point Cloud Augmentation (Chapter 7).

Cross-scan sector swapping in polar coordinates that preserves LiDAR scan line
continuity and angular consistency.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("lidarlens.augment.polar_mix")


def polar_mix(
    pts_a: Any = None,
    boxes_a: Any = None,
    labels_a: Any = None,
    pts_b: Any = None,
    boxes_b: Any = None,
    labels_b: Any = None,
    omega_range: Tuple[float, float] = (np.pi / 4.0, 3.0 * np.pi / 4.0),
    seed: Optional[int] = None,
    data_dict1: Optional[Dict[str, Any]] = None,
    data_dict2: Optional[Dict[str, Any]] = None,
    sector_angles: Optional[Tuple[float, float]] = None,
) -> Any:
    """Execute PolarMix sector swap between two scans (Chapter 7).

    Cuts an angular slice [phi, phi + omega] from Scene B and stitches it into
    Scene A, transferring points and 3D bounding boxes with azimuth consistency.
    Supports either dictionary inputs (data_dict1, data_dict2) or raw arrays.
    """
    is_dict_mode = False
    if isinstance(pts_a, dict) and data_dict1 is None:
        data_dict1 = pts_a
        data_dict2 = boxes_a
        is_dict_mode = True
    elif data_dict1 is not None and data_dict2 is not None:
        is_dict_mode = True

    if is_dict_mode:
        pts_a = data_dict1["points"]
        boxes_a = data_dict1.get("gt_boxes", data_dict1.get("gt_bboxes_3d", np.empty((0, 7), dtype=np.float32)))
        labels_a = data_dict1.get("gt_names", data_dict1.get("gt_labels_3d", data_dict1.get("gt_labels", np.empty(0))))
        pts_b = data_dict2["points"]
        boxes_b = data_dict2.get("gt_boxes", data_dict2.get("gt_bboxes_3d", np.empty((0, 7), dtype=np.float32)))
        labels_b = data_dict2.get("gt_names", data_dict2.get("gt_labels_3d", data_dict2.get("gt_labels", np.empty(0))))

    rng = np.random.default_rng(seed)

    # 1. Sample sector boundaries
    if sector_angles is not None:
        phi_start, phi_end = sector_angles
        omega = phi_end - phi_start
    else:
        omega = rng.uniform(omega_range[0], omega_range[1])
        phi_start = rng.uniform(-np.pi, np.pi - omega)
        phi_end = phi_start + omega

    # 2. Compute azimuth angles in [-pi, pi]
    az_a = np.arctan2(pts_a[:, 1], pts_a[:, 0])
    az_b = np.arctan2(pts_b[:, 1], pts_b[:, 0])

    in_sector_a = (az_a >= phi_start) & (az_a <= phi_end)
    in_sector_b = (az_b >= phi_start) & (az_b <= phi_end)

    # Keep points outside sector from A, and inside sector from B
    kept_pts_a = pts_a[~in_sector_a]
    swapped_pts_b = pts_b[in_sector_b]
    mixed_points = np.vstack([kept_pts_a, swapped_pts_b])

    # 3. Synchronize 3D bounding boxes
    kept_boxes_a: list = []
    kept_labels_a: list = []
    if len(boxes_a) > 0:
        box_az_a = np.arctan2(boxes_a[:, 1], boxes_a[:, 0])
        mask_box_a = ~((box_az_a >= phi_start) & (box_az_a <= phi_end))
        kept_boxes_a = [boxes_a[mask_box_a]]
        kept_labels_a = [list(np.array(labels_a)[mask_box_a])]

    kept_boxes_b: list = []
    kept_labels_b: list = []
    if len(boxes_b) > 0:
        box_az_b = np.arctan2(boxes_b[:, 1], boxes_b[:, 0])
        mask_box_b = (box_az_b >= phi_start) & (box_az_b <= phi_end)
        kept_boxes_b = [boxes_b[mask_box_b]]
        kept_labels_b = [list(np.array(labels_b)[mask_box_b])]

    all_boxes = [b for b in (kept_boxes_a + kept_boxes_b) if len(b) > 0]
    all_labels = []
    for l_list in (kept_labels_a + kept_labels_b):
        all_labels.extend(l_list)

    mixed_boxes = np.vstack(all_boxes) if all_boxes else np.empty((0, 7), dtype=np.float32)
    mixed_labels = np.array(all_labels) if all_labels else np.empty(0)

    meta = {
        "operator": "PolarMix",
        "phi_start_rad": float(phi_start),
        "phi_end_rad": float(phi_end),
        "omega_rad": float(omega),
        "donor_points_injected": int(np.sum(in_sector_b)),
    }

    if is_dict_mode:
        res = dict(data_dict1)
        res["points"] = mixed_points
        if "gt_boxes" in res:
            res["gt_boxes"] = mixed_boxes
        if "gt_bboxes_3d" in res:
            res["gt_bboxes_3d"] = mixed_boxes
        if "gt_names" in res:
            res["gt_names"] = list(mixed_labels)
        if "gt_labels" in res:
            res["gt_labels"] = mixed_labels
        res["aug_metadata"] = meta
        return res

    return mixed_points, mixed_boxes, mixed_labels, meta
