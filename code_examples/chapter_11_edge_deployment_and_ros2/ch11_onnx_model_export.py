"""
Chapter 11: Edge Deployment & ROS 2 Perception Systems
Code Example 11.1: PointPillars Backbone ONNX Computational Graph Export & Validation.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第11章 11.1节 (Para 13)
"""

import os
import tempfile
import time
import numpy as np
import torch
import onnx
import lidarlens as ll
from lidarlens.models import PointPillarsBackbone
from lidarlens.deploy import export_to_onnx


def main():
    print("=== Chapter 11: ONNX Computational Graph Export & Validation ===")

    # 1. Initialize PointPillars Backbone
    model = PointPillarsBackbone(in_channels=64, out_channels=256)
    model.eval()

    # Typical PointPillars pseudo-image shape: (B, C, H, W) = (1, 64, 496, 432)
    dummy_input = torch.randn(1, 64, 496, 432, dtype=torch.float32)

    with torch.no_grad():
        ref_output = model(dummy_input)

    print(f"PyTorch Model Architecture: PointPillarsBackbone (64 -> 128 -> 256 channels)")
    print(f"Exemplar Pseudo-Image Input Shape:  {tuple(dummy_input.shape)}")
    print(f"Reference High-Level Output Shape:  {tuple(ref_output.shape)}")

    # 2. Export Model to ONNX Graph
    temp_dir = tempfile.mkdtemp(prefix="lidarlens_onnx_")
    onnx_path = os.path.join(temp_dir, "pointpillars_backbone.onnx")

    t0 = time.perf_counter()
    exported_path = export_to_onnx(
        model=model,
        save_path=onnx_path,
        dummy_input=dummy_input,
        opset_version=14,
        device="cpu"
    )
    export_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nONNX Graph exported successfully in {export_ms:.2f} ms:")
    print(f"  Target File: {exported_path}")
    file_size_mb = os.path.getsize(exported_path) / (1024 * 1024)
    print(f"  Exported Binary Size: {file_size_mb:.2f} MB")

    # 3. Validate Graph Topology with ONNX Checker
    onnx_model = onnx.load(exported_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model Structure Validation: PASSED (Opset 14 compliant)")

    # 4. Verify Input/Output Metadata & Dynamic Axes
    graph = onnx_model.graph
    input_names = [inp.name for inp in graph.input]
    output_names = [out.name for out in graph.output]

    print(f"  Input Tensor Names:  {input_names}")
    print(f"  Output Tensor Names: {output_names}")

    assert "pillar_features" in input_names
    assert "spatial_features" in output_names

    # Check dynamic batch dimension
    batch_dim_param = graph.input[0].type.tensor_type.shape.dim[0]
    is_dynamic_batch = (batch_dim_param.dim_param == "batch_size") or (batch_dim_param.dim_value == 0)
    print(f"  Dynamic Batch Axis Configured: {is_dynamic_batch}")
    assert is_dynamic_batch, "Dynamic batch dimension was not properly exported!"

    # Clean up temporary exported file
    try:
        os.remove(exported_path)
        os.rmdir(temp_dir)
    except Exception:
        pass

    print("\nPointPillars Backbone ONNX deployment pipeline verified successfully.")
    print("Chapter 11.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
