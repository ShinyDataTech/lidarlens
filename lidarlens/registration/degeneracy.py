"""LidarLens Odometry Degeneracy Detection (Chapter 5).

Analyzes the 6x6 Gauss-Newton Hessian matrix to detect geometric degeneracy
in long corridors, tunnels, or featureless plains.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger("lidarlens.registration.degeneracy")


def detect_degeneracy(
    H: np.ndarray,
    thresh_lambda: float = 1.0,
    thresh_cond: float = 100.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    if "threshold" in kwargs:
        thresh_cond = float(kwargs["threshold"])
    if H.shape != (6, 6):
        raise ValueError(f"Hessian matrix must have shape (6, 6), got {H.shape}")

    # 1. Dimensional isolation: Decouple 3x3 rotation and 3x3 translation blocks
    H_rot = H[:3, :3]
    H_trans = H[3:, 3:]

    deg_info: Dict[str, Any] = {
        "rot": [],
        "trans": [],
        "degenerate_directions": [],
        "is_degenerate": False,
        "condition_number_rotation": 1.0,
        "condition_number_translation": 1.0,
    }

    for key, matrix in [("rot", H_rot), ("trans", H_trans)]:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = eigenvalues.argsort()
        val = eigenvalues[idx]
        vec = eigenvectors[:, idx]
        cond = float(val[-1] / (val[0] + 1e-9))
        
        if key == "rot":
            deg_info["condition_number_rotation"] = cond
        else:
            deg_info["condition_number_translation"] = cond

        if val[0] < thresh_lambda or cond > thresh_cond:
            deg_axis = vec[:, 0]
            deg_info[key].append(deg_axis)
            deg_info["degenerate_directions"].append(deg_axis)
            deg_info["is_degenerate"] = True
            logger.warning(
                "Geometric degeneracy detected in %s block! min_eig=%.4e, cond=%.2f, axis=%s",
                key,
                val[0],
                cond,
                deg_axis.round(3),
            )

    return deg_info
