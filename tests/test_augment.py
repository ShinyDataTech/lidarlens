"""Tests for 3D Data Augmentation & Quality Gates (Chapter 7)."""

import numpy as np
import pytest

from lidarlens.augment import AugmentationPipeline, polar_mix, mix3d, run_8point_quality_gates


class TestAugmentation:
    def test_augmentation_pipeline_seed_locking(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10, 10, size=(100, 4)).astype(np.float32)
        annos = {
            "bboxes": np.array([[2.0, 3.0, 0.0, 4.0, 1.8, 1.5, 0.5]], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }

        pipeline = AugmentationPipeline(cfg_or_path={"aug_strategy": []})

        # Run twice with same seed
        p1, b1, m1 = pipeline.process_sample(pts, annos, seed=2026)
        p2, b2, m2 = pipeline.process_sample(pts, annos, seed=2026)

        np.testing.assert_allclose(p1, p2)
        np.testing.assert_allclose(b1, b2)
        assert m1["seed"] == 2026
        assert len(m1["applied_ops"]) > 0

    def test_polar_mix(self):
        rng = np.random.default_rng(42)
        pts_a = rng.uniform(-10, 10, size=(200, 4)).astype(np.float32)
        boxes_a = np.array([[3.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32)
        labels_a = np.array([0], dtype=np.int64)

        pts_b = rng.uniform(-10, 10, size=(150, 4)).astype(np.float32)
        boxes_b = np.array([[-3.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32)
        labels_b = np.array([1], dtype=np.int64)

        mixed_pts, mixed_boxes, mixed_labels, meta = polar_mix(
            pts_a, boxes_a, labels_a, pts_b, boxes_b, labels_b, seed=42
        )

        assert mixed_pts.ndim == 2
        assert mixed_pts.shape[1] == 4
        assert "operator" in meta

    def test_mix3d(self):
        rng = np.random.default_rng(42)
        pts_a = rng.uniform(-5, 5, size=(100, 4)).astype(np.float32)
        boxes_a = np.array([[2.0, 0.0, 0.0, 2.0, 1.5, 1.2, 0.0]], dtype=np.float32)
        labels_a = np.array([0], dtype=np.int64)

        pts_b = rng.uniform(-5, 5, size=(100, 4)).astype(np.float32)
        boxes_b = np.array([[-3.0, 2.0, 0.0, 2.0, 1.5, 1.2, 0.0]], dtype=np.float32)
        labels_b = np.array([1], dtype=np.int64)

        mixed_pts, mixed_boxes, mixed_labels, meta = mix3d(
            pts_a, boxes_a, labels_a, pts_b, boxes_b, labels_b, collision_check=True
        )

        assert len(mixed_pts) == 200
        assert len(mixed_boxes) == 2
        assert len(mixed_labels) == 2


class TestQualityGates:
    def test_run_8point_quality_gates(self):
        rng = np.random.default_rng(42)
        # Create a valid synthetic LiDAR scan
        xyz = rng.uniform(-20, 20, size=(1000, 3)).astype(np.float32)
        xyz[:, 2] = rng.uniform(-1.5, 1.0, size=1000)  # Valid ground clearance
        intensity = rng.uniform(10, 200, size=(1000, 1)).astype(np.float32)
        points = np.hstack([xyz, intensity])

        # Valid 3D boxes
        bboxes = np.array([
            [5.0, 0.0, -0.5, 4.0, 1.8, 1.5, 0.0],
            [-8.0, 4.0, -0.5, 3.8, 1.8, 1.4, 0.5],
        ], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int64)

        report = run_8point_quality_gates(points, bboxes, labels)
        assert "passed" in report
        assert "checks" in report
        assert report["total_gates"] == 8
