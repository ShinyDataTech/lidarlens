"""
Chapter 9: Sim-to-Real Domain Adaptation & Test-Time Adaptation
Code Example 9.4: Test-Time Adaptation (Tent) via Shannon Entropy Minimization.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第9章 9.4节 (Para 83)
"""

import time
import torch
import torch.nn as nn
from lidarlens.sim2real import setup_tta_model, tta_step


class PerceptionClassifier(nn.Module):
    """Target classifier with BatchNorm layers suitable for test-time adaptation."""

    def __init__(self, in_features: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


def main():
    print("=== Chapter 9: Test-Time Adaptation (Tent) ===")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_model = PerceptionClassifier(in_features=32, num_classes=4).to(device)

    # 1. Setup Model for Test-Time Adaptation
    # Freezes all weights except BatchNorm affine parameters (gamma, beta)
    model = setup_tta_model(raw_model)

    # Verify parameter freezing contract
    for name, param in model.named_parameters():
        if "bn" in name and ("weight" in name or "bias" in name):
            assert param.requires_grad is True, f"{name} should be unfrozen for TTA!"
        else:
            assert param.requires_grad is False, f"{name} should remain frozen during TTA!"

    # Verify track_running_stats is disabled
    assert model.bn1.track_running_stats is False
    assert model.bn1.running_mean is None
    print("TTA Model Setup verified: Only BatchNorm affine parameters are active for gradient updates.")

    # Only pass unfrozen parameters to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=0.01)

    # 2. Simulate Out-of-Distribution (OOD) Streaming Data
    # A sudden environmental domain shift (e.g. heavy fog/rain shift) increases entropy
    streaming_frames = 20
    batch_size = 32
    initial_entropy = None
    final_entropy = None

    print(f"\nStreaming {streaming_frames} test frames through TTA loop:")
    t0 = time.perf_counter()

    for frame_idx in range(1, streaming_frames + 1):
        # OOD distribution with domain bias shift
        shifted_batch = torch.randn(batch_size, 32, device=device) * 2.0 + 1.5

        # Execute single-step entropy minimization
        outputs = tta_step(model, optimizer, shifted_batch)

        # Compute batch Shannon entropy
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean().item()

        if frame_idx == 1:
            initial_entropy = entropy

        if frame_idx % 5 == 0 or frame_idx == 1:
            print(f"  Frame [{frame_idx:>2}/{streaming_frames}] - Prediction Shannon Entropy H(p): {entropy:.4f}")

        final_entropy = entropy

    tta_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n{streaming_frames} TTA streaming adaptation frames processed in {tta_ms:.2f} ms ({tta_ms / streaming_frames:.2f} ms/frame)")

    # Assertions: Entropy must decrease as predictions become more confident and adapted
    print(f"Initial Entropy: {initial_entropy:.4f} -> Adapted Final Entropy: {final_entropy:.4f}")
    assert final_entropy < initial_entropy, f"Entropy did not decrease: {initial_entropy:.4f} -> {final_entropy:.4f}"
    print("Tent test-time adaptation algorithm verified successfully.")
    print("Chapter 9.4 verification completed successfully.\n")


if __name__ == "__main__":
    main()
