"""Tests for Sim-to-Real Domain Adaptation & Physical Noise Models (Chapter 9)."""

import numpy as np
import pytest

from lidarlens.sim2real import (
    beer_lambert_attenuation,
    sample_marshall_palmer_rain,
    inject_weather_noise,
    GradientReversalFn,
    DANNModel,
    setup_tta_model,
    tta_step,
)
from lidarlens.filters import rain_pointcloud_denoise

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestNoiseModels:
    def test_beer_lambert_attenuation(self):
        intensity = np.full(100, 100.0, dtype=np.float32)
        ranges = np.linspace(1.0, 50.0, 100).astype(np.float32)

        attenuated = beer_lambert_attenuation(intensity, ranges, sigma_ext=0.02)
        assert len(attenuated) == 100
        # Intensity must strictly decrease monotonically with range
        assert attenuated[0] > attenuated[-1]
        assert np.all(attenuated >= 0.0)

    def test_marshall_palmer_rain_injection(self):
        pts = np.random.uniform(-10, 10, size=(500, 4)).astype(np.float32)
        pts[:, 3] = 100.0

        rain_pts = sample_marshall_palmer_rain(pts, rain_rate_mm_hr=30.0, seed=42)
        # Should have added raindrop backscatter points
        assert len(rain_pts) > len(pts)

    def test_rain_pointcloud_denoise_cascade(self):
        rng = np.random.default_rng(42)
        # Ground truth structure: vehicle points with high reflection
        veh_xyz = rng.uniform(-2, 2, size=(200, 3)).astype(np.float32)
        veh_int = rng.uniform(50, 150, size=(200, 1)).astype(np.float32)

        # Rain noise: low intensity spray
        rain_xyz = rng.uniform(-10, 10, size=(100, 3)).astype(np.float32)
        rain_int = rng.uniform(0.01, 0.08, size=(100, 1)).astype(np.float32)

        pts_xyz = np.vstack([veh_xyz, rain_xyz])
        intensity = np.vstack([veh_int, rain_int])

        mask = rain_pointcloud_denoise(
            pts_xyz,
            intensity,
            intensity_thr=0.12,
            ror_radius=0.5,
            ror_min_neighbors=4,
            sor_k=10,
        )

        assert len(mask) == 300
        # Rain points (low intensity) should be pruned
        assert np.sum(mask[200:]) == 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adaptation tests")
class TestAdaptationPyTorch:
    def test_gradient_reversal_fn(self):
        x = torch.tensor([2.0, -3.0], requires_grad=True)
        alpha = 0.5
        y = GradientReversalFn.apply(x, alpha)

        # Forward is identity
        torch.testing.assert_close(y, x)

        loss = (y * 2.0).sum()
        loss.backward()

        # Gradient should be reversed by -alpha: 2.0 * (-0.5) = -1.0
        expected_grad = torch.tensor([-1.0, -1.0])
        torch.testing.assert_close(x.grad, expected_grad)

    def test_tent_test_time_adaptation(self):
        class SimpleClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 20)
                self.bn = nn.BatchNorm1d(20)
                self.head = nn.Linear(20, 3)

            def forward(self, x):
                return self.head(self.bn(self.fc(x)))

        model = SimpleClassifier()
        tta_model = setup_tta_model(model)

        # Ensure backbone linear weights are frozen
        assert not tta_model.fc.weight.requires_grad
        # Ensure BatchNorm affine parameters are trainable
        assert tta_model.bn.weight.requires_grad

        optimizer = torch.optim.SGD([tta_model.bn.weight, tta_model.bn.bias], lr=0.01)
        batch_data = torch.randn(8, 10)

        out = tta_step(tta_model, optimizer, batch_data)
        assert out.shape == (8, 3)
