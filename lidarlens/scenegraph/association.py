"""LidarLens Spatio-Temporal Association & Feature Fusion (Chapter 10).

Integrates multi-modal observation association combining Mahalanobis distance
gating with CLIP/DINOv2 appearance embedding cosine similarity and EWMA fusion.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from lidarlens.scenegraph.graph import SceneGraph3D, SceneNode

logger = logging.getLogger("lidarlens.scenegraph.association")


def compute_mahalanobis_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    cov_diag: np.ndarray = np.array([0.25, 0.25, 0.25], dtype=np.float32),
) -> float:
    """Compute Mahalanobis distance between two 3D spatial points."""
    diff = p1 - p2
    inv_cov = 1.0 / (cov_diag + 1e-6)
    return float(np.sqrt(np.sum(diff * diff * inv_cov)))


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two feature embedding vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def associate_and_update(
    new_obs: SceneNode,
    graph: SceneGraph3D,
    mahalanobis_thresh: float = 3.0,
    cosine_thresh: float = 0.85,
    alpha: float = 0.2,
) -> Tuple[SceneNode, bool]:
    """Associate new spatial observation with existing scene graph nodes (Chapter 10).

    Args:
        new_obs: Newly perceived SceneNode instance.
        graph: Active SceneGraph3D.
        mahalanobis_thresh: Spatial gating threshold.
        cosine_thresh: Minimum CLIP/DINOv2 semantic cosine similarity.
        alpha: Exponentially Weighted Moving Average (EWMA) semantic update rate.

    Returns:
        Tuple of (target_node, is_matched).
    """
    # 1. Spatial Candidate Gating
    spatial_candidates = []
    for node in graph.nodes.values():
        if node.layer != new_obs.layer:
            continue
        dist_m = compute_mahalanobis_distance(new_obs.pose, node.pose)
        if dist_m <= mahalanobis_thresh:
            spatial_candidates.append(node)

    # 2. Appearance Association
    best_match: Optional[SceneNode] = None
    max_sim = 0.0

    if new_obs.appearance_feat is not None:
        for node in spatial_candidates:
            if node.appearance_feat is not None:
                sim = compute_cosine_similarity(new_obs.appearance_feat, node.appearance_feat)
                if sim > max_sim and sim >= cosine_thresh:
                    max_sim = sim
                    best_match = node
    else:
        # Fallback to closest spatial candidate of matching label
        for node in spatial_candidates:
            if node.label.lower() == new_obs.label.lower():
                best_match = node
                break

    # 3. Fusion Strategy
    if best_match is not None:
        # A. Kalman / weighted pose update
        best_match.pose = (0.7 * best_match.pose + 0.3 * new_obs.pose).astype(np.float32)
        best_match.extent = (0.8 * best_match.extent + 0.2 * new_obs.extent).astype(np.float32)

        # B. EWMA semantic feature update
        if new_obs.appearance_feat is not None and best_match.appearance_feat is not None:
            best_match.appearance_feat = (
                (1.0 - alpha) * best_match.appearance_feat + alpha * new_obs.appearance_feat
            ).astype(np.float32)

        # C. Confidence accumulation
        best_match.confidence = min(1.0, best_match.confidence + 0.1)

        logger.info("Associated new observation to existing node %s (sim=%.2f)", best_match.id, max_sim)
        return best_match, True
    else:
        # 4. Create new instance node in graph
        graph.add_node(new_obs)
        logger.info("Created new node %s in 3D Scene Graph.", new_obs.id)
        return new_obs, False
