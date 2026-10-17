"""LidarLens Mix3D Scene Superposition Augmentation (Chapter 7).

Merges two full 3D point cloud scenes and performs 3D bounding box collision audits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("lidarlens.augment.mix3d")


def compute_box_bev_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """Check if two 3D bounding boxes collide in 2D BEV projection."""
    # Approximate with 2D circular radius or AABB overlap
    dist = np.linalg.norm(box1[:2] - box2[:2])
    r1 = 0.5 * np.hypot(box1[3], box1[4])
    r2 = 0.5 * np.hypot(box2[3], box2[4])
    return bool(dist < (r1 + r2) * 0.8)


def mix3d(
    pts_a: Any = None,
    boxes_a: Any = None,
    labels_a: Any = None,
    pts_b: Any = None,
    boxes_b: Any = None,
    labels_b: Any = None,
    collision_check: bool = True,
    seed: Optional[int] = None,
    data_dict1: Optional[Dict[str, Any]] = None,
    data_dict2: Optional[Dict[str, Any]] = None,
) -> Any:
    """Superimpose Scene B into Scene A with instance collision prevention.

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

    merged_pts = [pts_a]
    merged_boxes = [boxes_a] if len(boxes_a) > 0 else []
    merged_labels = [list(labels_a)] if len(labels_a) > 0 else []

    kept_donor_box_indices = []

    if len(boxes_b) > 0:
        for idx_b, b_box in enumerate(boxes_b):
            collides = False
            if collision_check and len(boxes_a) > 0:
                for a_box in boxes_a:
                    if compute_box_bev_overlap(a_box, b_box):
                        collides = True
                        break
            if not collides:
                kept_donor_box_indices.append(idx_b)

    if kept_donor_box_indices:
        b_kept_boxes = boxes_b[kept_donor_box_indices]
        b_kept_labels = list(np.array(labels_b)[kept_donor_box_indices])
        merged_boxes.append(b_kept_boxes)
        merged_labels.append(b_kept_labels)

    merged_pts.append(pts_b)

    final_points = np.vstack(merged_pts)
    final_boxes = np.vstack(merged_boxes) if merged_boxes else np.empty((0, 7), dtype=np.float32)
    all_labels = []
    for l_list in merged_labels:
        all_labels.extend(l_list)
    final_labels = np.array(all_labels) if all_labels else np.empty(0)

    meta = {
        "operator": "Mix3D",
        "primary_points": len(pts_a),
        "donor_points_added": len(pts_b),
        "donor_boxes_retained": len(kept_donor_box_indices),
    }

    if is_dict_mode:
        res = dict(data_dict1)
        res["points"] = final_points
        if "gt_boxes" in res:
            res["gt_boxes"] = final_boxes
        if "gt_bboxes_3d" in res:
            res["gt_bboxes_3d"] = final_boxes
        if "gt_names" in res:
            res["gt_names"] = list(final_labels)
        if "gt_labels" in res:
            res["gt_labels"] = final_labels
        res["aug_metadata"] = meta
        return res

    return final_points, final_boxes, final_labels, meta
