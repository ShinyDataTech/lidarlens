"""LidarLens ONNX Model Export & Graph Optimization (Chapters 8 & 11).

Exports 3D perception backbones to standardized ONNX computational graphs
with dynamic batch dimension annotations and Opset 14 operator fidelity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("lidarlens.deploy.onnx_export")

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = object  # type: ignore

_HAS_ONNX = False
try:
    import onnx  # type: ignore
    _HAS_ONNX = True
except ImportError:
    onnx = None


def export_to_onnx(
    model: Any,
    save_path: str = "pointpillars_backbone.onnx",
    dummy_input: Optional[Any] = None,
    opset_version: int = 14,
    device: str = "cpu",
) -> str:
    """Export PyTorch 3D perception model to ONNX with dynamic batch axes (Chapter 11).

    Args:
        model: PyTorch nn.Module instance.
        save_path: Destination path for .onnx file.
        dummy_input: Optional exemplar input tensor (B, C, H, W).
        opset_version: Target ONNX Opset (default 14).
        device: Device to place dummy tensor on ('cpu' or 'cuda').

    Returns:
        Absolute file path of exported ONNX graph.
    """
    if not _HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required for export_to_onnx.")

    if not _HAS_ONNX:
        raise ImportError("The 'onnx' package is required to export ONNX models. Install via 'pip install onnx'.")

    model.eval()

    if dummy_input is None:
        # Standard PointPillars pseudo-image shape: (B, C, H, W) = (1, 64, 496, 432)
        dummy_input = torch.randn(1, 64, 496, 432, dtype=torch.float32)

    logger.info("Exporting ONNX graph to %s with Opset %d...", save_path, opset_version)

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        kwargs: dict = {
            "export_params": True,
            "opset_version": opset_version,
            "do_constant_folding": True,
            "input_names": ["pillar_features"],
            "output_names": ["spatial_features"],
            "dynamic_axes": {
                "pillar_features": {0: "batch_size"},
                "spatial_features": {0: "batch_size"},
            },
        }
        # Avoid onnxscript requirement on PyTorch 2.x by setting dynamo=False if supported
        try:
            torch.onnx.export(model, dummy_input, str(save_path), dynamo=False, **kwargs)
        except (TypeError, ModuleNotFoundError):
            torch.onnx.export(model, dummy_input, str(save_path), **kwargs)
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise

    logger.info("Successfully exported ONNX model to %s", save_path)
    return str(save_path)
