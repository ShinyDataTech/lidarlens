"""Tests for 3D Target Detection Models & ONNX Deployment (Chapters 8 & 11)."""

from pathlib import Path
import numpy as np
import pytest

from lidarlens.models import PointPillarsBackbone, pillarize_points
from lidarlens.deploy import export_to_onnx
from lidarlens.dataset import check_kitti_subset

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestOpsAndDataset:
    def test_pillarize_points(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform([0, -20, -2, 0], [40, 20, 0, 100], size=(1000, 4)).astype(np.float32)

        p_feats, p_coords, n_pts = pillarize_points(
            pts,
            voxel_size=(0.2, 0.2, 4.0),
            point_cloud_range=(0.0, -30.0, -3.0, 50.0, 30.0, 1.0),
            max_points_per_pillar=16,
            max_pillars=500,
        )

        assert p_feats.shape == (500, 16, 9)
        assert p_coords.shape == (500, 2)
        assert len(n_pts) == 500
        assert np.sum(n_pts > 0) > 0

    def test_check_kitti_subset(self, tmp_path):
        # Create minimal dummy KITTI structure
        v_dir = tmp_path / "velodyne"
        l_dir = tmp_path / "label_2"
        v_dir.mkdir()
        l_dir.mkdir()

        # Write dummy bin and label
        dummy_pts = np.random.randn(2000, 4).astype(np.float32)
        dummy_pts.tofile(str(v_dir / "000000.bin"))

        with open(l_dir / "000000.txt", "w") as f:
            f.write("Car 0.0 0 0.0 0.0 0.0 0.0 0.0 1.5 1.8 4.0 2.0 1.0 10.0 0.0\n")

        report = check_kitti_subset(str(tmp_path))
        assert report["num_samples"] == 1
        assert report["total_cars"] == 1
        assert report["avg_points"] == 2000


try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for PointPillars")
class TestPointPillarsAndONNX:
    def test_pointpillars_backbone_forward(self):
        model = PointPillarsBackbone(in_channels=64, out_channels=256)
        x = torch.randn(1, 64, 100, 100)
        out = model(x)
        assert out.shape == (1, 256, 100, 100)

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx package required for ONNX export")
    def test_export_to_onnx(self, tmp_path):
        model = PointPillarsBackbone(in_channels=64, out_channels=256)
        save_file = str(tmp_path / "pointpillars.onnx")
        dummy_input = torch.randn(1, 64, 64, 64)

        onnx_path = export_to_onnx(model, save_path=save_file, dummy_input=dummy_input, opset_version=14)

        assert Path(onnx_path).is_file()
        assert Path(onnx_path).stat().st_size > 1000
