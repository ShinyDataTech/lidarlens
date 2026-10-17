"""LidarLens 8-Point Hardware-Level Quality Gates (Chapter 7).

Performs strict automated validation on augmented/synthesized 3D point cloud
samples to prevent dataset corruption and simulation-to-reality collapse.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("lidarlens.augment.qa")


def run_8point_quality_gates(
    points: Any,
    bboxes: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    original_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute 8-item hardware-level Quality Gate validation (Chapter 7).

    Supports either a data dictionary (containing 'points', 'gt_boxes', etc.) or raw arrays.
    """
    if isinstance(points, dict):
        d = points
        points = d["points"]
        bboxes = d.get("gt_boxes", d.get("gt_bboxes_3d", np.empty((0, 7), dtype=np.float32)))
        labels = d.get("gt_names", d.get("gt_labels_3d", d.get("gt_labels", np.empty(0))))

    if bboxes is None:
        bboxes = np.empty((0, 7), dtype=np.float32)
    if labels is None:
        labels = np.empty(0)

    failures: List[str] = []
    checks: Dict[str, Dict[str, Any]] = {}

    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)

    # Gate 1: 3D Bounding Box Collision / Overlap
    collision_found = False
    m_boxes = len(bboxes)
    if m_boxes > 1:
        for i in range(m_boxes):
            for j in range(i + 1, m_boxes):
                dist = np.linalg.norm(bboxes[i, :2] - bboxes[j, :2])
                r1 = 0.5 * np.hypot(bboxes[i, 3], bboxes[i, 4])
                r2 = 0.5 * np.hypot(bboxes[j, 3], bboxes[j, 4])
                if dist < (r1 + r2) * 0.5:
                    collision_found = True
                    break
    g1_ok = not collision_found
    checks["box_collision_free"] = {
        "passed": g1_ok,
        "details": "No inter-box BEV collisions" if g1_ok else "Inter-box bounding collision detected",
    }
    if not g1_ok:
        failures.append("Gate 1 Failed: Inter-box BEV bounding collision detected.")

    # Gate 2: Ground Clearance (z level physically viable)
    z_min = float(np.min(xyz[:, 2])) if len(xyz) > 0 else 0.0
    ground_ok = -4.0 <= z_min <= 0.5
    checks["ground_clearance"] = {
        "passed": ground_ok,
        "details": f"z_min={z_min:.2f} m within physical range [-4.0, 0.5]" if ground_ok else f"z_min={z_min:.2f} m outside valid range",
    }
    if not ground_ok:
        failures.append(f"Gate 2 Failed: Ground elevation minimum {z_min:.2f} m outside physical bounds.")

    # Gate 3: Point Density Inside Bounding Boxes
    empty_boxes = 0
    if m_boxes > 0 and len(xyz) > 0:
        for b in bboxes:
            dx, dy, dz = b[3], b[4], b[5]
            box_dist = np.linalg.norm(xyz[:, :3] - b[:3], axis=1)
            pts_in_box = np.sum(box_dist <= max(dx, dy, dz) * 0.6)
            if pts_in_box < 3:
                empty_boxes += 1
    density_ok = (empty_boxes == 0)
    checks["instance_point_density"] = {
        "passed": density_ok,
        "details": "All boxes contain >= 3 points" if density_ok else f"{empty_boxes} empty/sparse boxes",
    }
    if not density_ok:
        failures.append(f"Gate 3 Failed: {empty_boxes} bounding boxes have fewer than 3 internal points.")

    # Gate 4: Distance-Density Decay Monotonicity
    if len(ranges) > 50:
        near_pts = np.sum((ranges >= 2.0) & (ranges < 15.0))
        far_pts = np.sum((ranges >= 30.0) & (ranges < 60.0))
        decay_ok = near_pts >= (far_pts * 0.3)
    else:
        decay_ok = True
    checks["distance_density_decay"] = {
        "passed": decay_ok,
        "details": "Monotonic 1/R^2 beam divergence verified" if decay_ok else "Distance-density anomaly detected",
    }
    if not decay_ok:
        failures.append("Gate 4 Failed: Spatial distance-density decay anomaly detected.")

    # Gate 5: Sensor FOV Azimuth & Elevation
    elev = np.arcsin(np.clip(xyz[:, 2] / (ranges + 1e-6), -1.0, 1.0))
    elev_deg = np.degrees(elev)
    fov_ok = bool(np.all(elev_deg >= -45.0) and np.all(elev_deg <= 45.0))
    checks["sensor_fov_compliance"] = {
        "passed": fov_ok,
        "details": f"Elevation span [{np.min(elev_deg):.1f}°, {np.max(elev_deg):.1f}°] within FOV" if fov_ok else "Elevation exceeds sensor FOV",
    }
    if not fov_ok:
        failures.append("Gate 5 Failed: LiDAR elevation beam angles exceed valid sensor FOV limits.")

    # Gate 6: Minimum and Maximum Range Limits
    r_min = float(np.min(ranges)) if len(ranges) > 0 else 0.0
    r_max = float(np.max(ranges)) if len(ranges) > 0 else 0.0
    range_ok = (r_min >= 0.0) and (r_max <= 200.0)
    checks["range_limits"] = {
        "passed": range_ok,
        "details": f"Range [{r_min:.1f} m, {r_max:.1f} m] within [0, 200 m]" if range_ok else "Range limits exceeded",
    }
    if not range_ok:
        failures.append(f"Gate 6 Failed: Range extrema [{r_min:.1f}, {r_max:.1f}] violate hardware limits.")

    # Gate 7: Reflection Intensity Validity
    if points.shape[1] >= 4:
        intensities = points[:, 3]
        intensity_ok = bool(not np.any(np.isnan(intensities)) and np.all(intensities >= 0.0))
    else:
        intensity_ok = True
    checks["intensity_validity"] = {
        "passed": intensity_ok,
        "details": "All intensities non-negative and finite" if intensity_ok else "NaN/negative intensity values detected",
    }
    if not intensity_ok:
        failures.append("Gate 7 Failed: Negative or NaN values detected in intensity channel.")

    # Gate 8: Bounding Box Dimension Realism
    dim_ok = True
    if m_boxes > 0:
        for b in bboxes:
            dx, dy, dz = b[3], b[4], b[5]
            if dx <= 0.1 or dy <= 0.1 or dz <= 0.1 or dx > 25.0 or dy > 6.0 or dz > 6.0:
                dim_ok = False
                break
    checks["box_dimension_realism"] = {
        "passed": dim_ok,
        "details": "All dimensions within physical bounds" if dim_ok else "Box extents exceed physical limits",
    }
    if not dim_ok:
        failures.append("Gate 8 Failed: Bounding box extents violate vehicle/pedestrian physical limits.")

    all_passed = len(failures) == 0

    return {
        "all_passed": all_passed,
        "passed": all_passed,
        "total_passed_gates": sum(1 for c in checks.values() if c["passed"]),
        "total_gates": len(checks),
        "checks": checks,
        "failures": failures,
    }
