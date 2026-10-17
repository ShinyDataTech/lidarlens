"""LidarLens 3D Data Augmentation Pipeline (Chapter 7).

Manages reproducible, config-driven 3D point cloud augmentation pipelines with
synchronized 3D bounding box coordinate and heading transformations.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from lidarlens.augment.polar_mix import polar_mix
from lidarlens.augment.mix3d import mix3d

logger = logging.getLogger("lidarlens.augment.pipeline")


class AugmentationPipeline:
    """Config-driven 3D point cloud augmentation pipeline (Chapter 7).

    Guarantees deterministic MLOps reproducibility via seed locking and maintains
    a complete audit trail of applied operators in aug_metadata.
    """

    def __init__(self, cfg_or_path: Union[Dict[str, Any], str, Path]) -> None:
        if isinstance(cfg_or_path, (str, Path)):
            p = Path(cfg_or_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self.cfg = yaml.safe_load(f)
            else:
                self.cfg = {"aug_strategy": []}
        else:
            self.cfg = dict(cfg_or_path)

        self.transforms = self.cfg.get("transforms", self.cfg.get("aug_strategy", []))
        self.seed = self.cfg.get("seed", 42)

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Callable interface matching MMDetection3D / PyTorch Dataset transforms."""
        return self.forward(data_dict)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass applying registered augmentations to a data dictionary."""
        res = dict(data_dict)
        points = res["points"].copy()
        boxes = res.get("gt_boxes", res.get("gt_bboxes_3d", None))
        if boxes is not None:
            boxes = boxes.copy()

        seed = self.seed
        rng = np.random.default_rng(seed)

        applied_ops = []

        for tf in self.transforms:
            op_type = tf.get("type", "")
            prob = tf.get("prob", 1.0)
            if rng.uniform(0.0, 1.0) > prob:
                continue

            if op_type == "GlobalRotScaleTrans":
                rot_range = tf.get("rot_range", [-np.pi / 8.0, np.pi / 8.0])
                angle = rng.uniform(rot_range[0], rot_range[1])
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

                scale_range = tf.get("scale_ratio_range", [0.95, 1.05])
                scale = rng.uniform(scale_range[0], scale_range[1])

                trans_std = tf.get("translation_std", [0.2, 0.2, 0.1])
                trans = rng.normal(0.0, trans_std).astype(np.float32)

                points[:, :3] = (points[:, :3] @ R.T) * scale + trans
                if boxes is not None and len(boxes) > 0:
                    boxes[:, :3] = (boxes[:, :3] @ R.T) * scale + trans
                    boxes[:, 3:6] *= scale
                    boxes[:, 6] += angle
                applied_ops.append(f"GlobalRotScaleTrans(scale={scale:.3f}, rot={np.rad2deg(angle):.1f}°)")

            elif op_type == "RandomFlip3D":
                p_flip = tf.get("flip_ratio_bev_horizontal", 0.5)
                if rng.uniform(0.0, 1.0) < p_flip:
                    points[:, 1] = -points[:, 1]
                    if boxes is not None and len(boxes) > 0:
                        boxes[:, 1] = -boxes[:, 1]
                        boxes[:, 6] = -boxes[:, 6]
                    applied_ops.append("RandomFlip3D(y_flip=True)")

            elif op_type == "GlobalTranslate":
                trans_std = tf.get("std", [0.2, 0.2, 0.0])
                shift = rng.normal(0.0, trans_std).astype(np.float32)
                points[:, :3] += shift
                if boxes is not None and len(boxes) > 0:
                    boxes[:, :3] += shift
                applied_ops.append("GlobalTranslate")

            elif op_type == "GlobalRotate":
                rot_range = tf.get("range", [-np.pi / 4.0, np.pi / 4.0])
                angle = rng.uniform(rot_range[0], rot_range[1])
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                points[:, :3] = points[:, :3] @ R.T
                if boxes is not None and len(boxes) > 0:
                    boxes[:, :3] = boxes[:, :3] @ R.T
                    boxes[:, 6] += angle
                applied_ops.append("GlobalRotate")

            elif op_type == "GlobalScale":
                scale_range = tf.get("range", [0.95, 1.05])
                scale = rng.uniform(scale_range[0], scale_range[1])
                points[:, :3] *= scale
                if boxes is not None and len(boxes) > 0:
                    boxes[:, :6] *= scale
                applied_ops.append("GlobalScale")

        res["points"] = points
        if "gt_boxes" in res:
            res["gt_boxes"] = boxes
        if "gt_bboxes_3d" in res:
            res["gt_bboxes_3d"] = boxes
        res["aug_metadata"] = {"applied_ops": applied_ops, "seed": seed}
        return res

    def process_sample(
        self,
        points: np.ndarray,
        annos: Dict[str, np.ndarray],
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Process a single point cloud sample with synchronized 3D bounding box transforms."""
        data_dict = {
            "points": points,
            "gt_boxes": annos.get("bboxes", np.empty((0, 7), dtype=np.float32)),
            "gt_labels": annos.get("labels", np.empty(0, dtype=np.int64)),
        }
        old_seed = self.seed
        self.seed = seed
        out = self.forward(data_dict)
        self.seed = old_seed
        return out["points"], out["gt_boxes"], out["aug_metadata"]
