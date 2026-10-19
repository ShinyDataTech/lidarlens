"""
Chapter 8: 3D Object Detection Training & Deep Learning
Code Example 8.3: Automatic Mixed Precision (AMP) 3D Detection Training Pipeline.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第8章 8.3 & 8.4节 (Paras 53 & 169)
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple
from lidarlens.models import PointPillarsBackbone


class PointPillarsDetectionHead(nn.Module):
    """Detection head predicting 3D bounding boxes and classification heatmaps."""

    def __init__(self, in_channels: int = 256, num_classes: int = 3, num_box_params: int = 7) -> None:
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, num_box_params, kernel_size=1)
        self.dir_head = nn.Conv2d(in_channels, 2, kernel_size=1) # Binary heading direction classification

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_preds = self.cls_head(x)
        reg_preds = self.reg_head(x)
        dir_preds = self.dir_head(x)
        return cls_preds, reg_preds, dir_preds


class Full3DDetector(nn.Module):
    """Integrated 3D detector model combining backbone and multi-task perception head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PointPillarsBackbone(in_channels=64, out_channels=256)
        self.head = PointPillarsDetectionHead(in_channels=256, num_classes=3, num_box_params=7)

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        pseudo_images = batch_dict["pseudo_images"] # (B, 64, H, W)
        bev_feats = self.backbone(pseudo_images)
        cls_preds, reg_preds, dir_preds = self.head(bev_feats)

        # Compute multi-task losses against targets
        cls_targets = batch_dict["cls_targets"]
        reg_targets = batch_dict["reg_targets"]
        dir_targets = batch_dict["dir_targets"]

        cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_preds, cls_targets)
        reg_loss = nn.functional.smooth_l1_loss(reg_preds, reg_targets)
        dir_loss = nn.functional.cross_entropy(dir_preds, dir_targets)

        total_loss = cls_loss + 2.0 * reg_loss + 0.2 * dir_loss

        loss_dict = {
            "c_loss": float(cls_loss.item()),
            "r_loss": float(reg_loss.item()),
            "d_loss": float(dir_loss.item()),
            "total_loss": float(total_loss.item())
        }
        pred_dict = {"cls": cls_preds, "reg": reg_preds, "dir": dir_preds}
        return total_loss, loss_dict, pred_dict


def generate_synthetic_batch(
    device: torch.device,
    batch_size: int = 2,
    grid_h: int = 64,
    grid_w: int = 64
) -> Dict[str, torch.Tensor]:
    """Create a mock mini-batch for training verification."""
    pseudo_images = torch.randn(batch_size, 64, grid_h, grid_w, device=device)
    cls_targets = torch.randint(0, 2, (batch_size, 3, grid_h, grid_w), device=device).float()
    reg_targets = torch.randn(batch_size, 7, grid_h, grid_w, device=device)
    dir_targets = torch.randint(0, 2, (batch_size, grid_h, grid_w), device=device).long()

    return {
        "pseudo_images": pseudo_images,
        "cls_targets": cls_targets,
        "reg_targets": reg_targets,
        "dir_targets": dir_targets,
    }


def main():
    print("=== Chapter 8: Automatic Mixed Precision (AMP) 3D Detection Training ===")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device} | CUDA Available: {use_cuda}")

    torch.manual_seed(42)

    # Instantiate model & optimizer
    model = Full3DDetector().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)

    # Initialize GradScaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"Optimizer: AdamW (lr=0.003, weight_decay=0.01)")
    print(f"AMP GradScaler enabled: {scaler.is_enabled()}")

    num_iterations = 15
    losses = []

    # Fixed mock batch to verify loss convergence
    batch = generate_synthetic_batch(device, batch_size=2, grid_h=64, grid_w=64)

    t0 = time.perf_counter()
    for i in range(1, num_iterations + 1):
        optimizer.zero_grad()

        # 1. Forward Pass under AMP context
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            total_loss, loss_dict, _ = model(batch)

        # 2. Backward Pass with Gradient Scaling
        scaler.scale(total_loss).backward()

        # 3. Optimizer Step & Scaler Update
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss_dict["total_loss"])

        if i % 5 == 0 or i == 1:
            print(
                f"Iteration [{i:>2}/{num_iterations}] - "
                f"Total Loss: {loss_dict['total_loss']:.4f} "
                f"(Cls: {loss_dict['c_loss']:.4f}, Reg: {loss_dict['r_loss']:.4f}, Dir: {loss_dict['d_loss']:.4f})"
            )

    train_ms = (time.perf_counter() - t0) * 1000.0
    avg_iter_ms = train_ms / num_iterations
    print(f"\n{num_iterations} iterations executed in {train_ms:.2f} ms ({avg_iter_ms:.2f} ms/iter)")

    # Assertions: Loss must decrease and remain numerically stable
    assert not torch.isnan(torch.tensor(losses)).any(), "NaN detected in training loss!"
    assert losses[-1] < losses[0], f"Loss failed to decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    print(f"Loss successfully converged from {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("Chapter 8.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
