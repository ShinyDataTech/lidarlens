"""LidarLens Sim-to-Real Domain Adaptation & Test-Time Adaptation (Chapter 9).

Provides GradientReversalFn for adversarial DANN domain alignment and Tent (Test-Time
Adaptation) entropy minimization for streaming test-time adaptation.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger("lidarlens.sim2real.adaptation")

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = object  # type: ignore
    Function = object  # type: ignore


if _HAS_TORCH:

    class GradientReversalFn(Function):
        """DANN Gradient Reversal Layer (GRL) core PyTorch autograd Function.

        Acts as identity mapping during forward inference, and multiplies gradients
        by -alpha during backpropagation to fool the domain discriminator.
        """

        @staticmethod
        def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            # Scale reversed gradient by -alpha
            return grad_output.neg() * ctx.alpha, None

    class DANNModel(nn.Module):
        """DANN adversarial domain adaptation network framework for 3D perception."""

        def __init__(
            self,
            feature_extractor: nn.Module,
            label_predictor: nn.Module,
            domain_classifier: nn.Module,
        ) -> None:
            super().__init__()
            self.feature_extractor = feature_extractor
            self.label_predictor = label_predictor
            self.domain_classifier = domain_classifier

        def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: Input point feature tensor.
                alpha: Gradient reversal scaling weight.

            Returns:
                Tuple of (class_prediction, domain_prediction).
            """
            features = self.feature_extractor(x)
            class_output = self.label_predictor(features)

            # Adversarial domain classification branch with reversed gradients
            reversed_features = GradientReversalFn.apply(features, alpha)
            domain_output = self.domain_classifier(reversed_features)

            return class_output, domain_output

    def setup_tta_model(model: nn.Module) -> nn.Module:
        """Configure model for streaming Test-Time Adaptation (Tent) (Chapter 9).

        Freezes all backbone weights while unfreezing only BatchNorm affine
        scale and shift parameters (gamma and beta). Disables running statistics
        accumulation to adapt strictly to streaming distribution shifts.
        """
        model.eval()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model

    def tta_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_data: torch.Tensor,
    ) -> torch.Tensor:
        """Execute single-step Shannon entropy minimization on unlabeled test stream."""
        optimizer.zero_grad()
        outputs = model(batch_data)

        # Shannon entropy: H(p) = -sum(p * log(p))
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()

        entropy.backward()
        optimizer.step()

        return outputs

else:

    class GradientReversalFn:  # type: ignore
        pass

    class DANNModel:  # type: ignore
        pass

    def setup_tta_model(model):  # type: ignore
        raise ImportError("PyTorch is required for setup_tta_model.")

    def tta_step(model, optimizer, batch_data):  # type: ignore
        raise ImportError("PyTorch is required for tta_step.")
