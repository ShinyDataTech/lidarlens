"""
Chapter 9: Sim-to-Real Domain Adaptation & Test-Time Adaptation
Code Example 9.3: Domain-Adversarial Neural Network (DANN) with Gradient Reversal Layer.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第9章 9.3节 (Para 54)
"""

import time
import torch
import torch.nn as nn
from typing import Tuple
from lidarlens.sim2real import GradientReversalFn, DANNModel


class FeatureExtractor(nn.Module):
    """3D point cloud spatial feature extractor."""

    def __init__(self, in_features: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LabelPredictor(nn.Module):
    """Primary 3D perception classification task branch."""

    def __init__(self, in_features: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DomainClassifier(nn.Module):
    """Adversarial domain discriminator (Source vs Target domain)."""

    def __init__(self, in_features: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) # Binary domain logits (0: Source/Synthetic, 1: Target/Real)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    print("=== Chapter 9: DANN Adversarial Domain Adaptation ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    fe = FeatureExtractor(in_features=64, hidden_dim=128)
    lp = LabelPredictor(in_features=128, num_classes=3)
    dc = DomainClassifier(in_features=128)
    
    dann = DANNModel(feature_extractor=fe, label_predictor=lp, domain_classifier=dc).to(device)
    dann.train()
    
    optimizer = torch.optim.Adam(dann.parameters(), lr=0.005)
    criterion_task = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    # 1. Verify Gradient Reversal Layer autograd inversion
    dummy_x = torch.randn(8, 64, device=device, requires_grad=True)
    alpha = 0.75
    
    # Forward through GRL directly
    grl_out = GradientReversalFn.apply(dummy_x, alpha)
    # Forward check: GRL should be strictly identity mapping during forward pass
    torch.testing.assert_close(grl_out, dummy_x)
    print(f"Gradient Reversal Layer forward identity check: PASSED")
    
    # Backward check: Gradient should be scaled by -alpha
    dummy_loss = grl_out.sum()
    dummy_loss.backward()
    expected_grad = -alpha * torch.ones_like(dummy_x)
    torch.testing.assert_close(dummy_x.grad, expected_grad)
    print(f"Gradient Reversal Layer backward gradient negation check: PASSED (Scale = -{alpha})")
    
    # 2. Simulate Domain Adaptation Step
    batch_size = 16
    # Source domain data (Synthetic simulation, labeled)
    src_x = torch.randn(batch_size, 64, device=device)
    src_y = torch.randint(0, 3, (batch_size,), device=device)
    src_domain = torch.zeros(batch_size, dtype=torch.long, device=device) # Domain 0
    
    # Target domain data (Real rainy environment, unlabeled for task)
    tgt_x = torch.randn(batch_size, 64, device=device) + 0.5 # Domain shift
    tgt_domain = torch.ones(batch_size, dtype=torch.long, device=device) # Domain 1
    
    print("\nSimulating DANN adversarial domain alignment loop:")
    t0 = time.perf_counter()
    for step in range(1, 11):
        optimizer.zero_grad()
        
        # Forward Source
        src_preds, src_d_preds = dann(src_x, alpha=alpha)
        loss_task = criterion_task(src_preds, src_y)
        loss_d_src = criterion_domain(src_d_preds, src_domain)
        
        # Forward Target
        _, tgt_d_preds = dann(tgt_x, alpha=alpha)
        loss_d_tgt = criterion_domain(tgt_d_preds, tgt_domain)
        
        total_loss = loss_task + 0.5 * (loss_d_src + loss_d_tgt)
        total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0 or step == 1:
            print(
                f"  Step [{step:>2}/10] - Task Loss: {loss_task.item():.4f} | "
                f"Domain Loss: {(loss_d_src + loss_d_tgt).item():.4f} | "
                f"Total Loss: {total_loss.item():.4f}"
            )
            
    loop_ms = (time.perf_counter() - t0) * 1000.0
    print(f"10 DANN adaptation iterations completed in {loop_ms:.2f} ms")
    
    assert total_loss.item() > 0.0 and not torch.isnan(total_loss)
    print("Chapter 9.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
