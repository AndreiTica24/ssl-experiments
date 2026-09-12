"""SimCLR model: ResNet-18 backbone + MLP projection head.

Architecture:
    image (3, 32, 32)
        │
        ▼
    ResNet-18 encoder (adapted for 32x32)
        │
        ▼
    features h (512-dim)  ← this is what we keep for downstream tasks
        │
        ▼
    Projection head: Linear → ReLU → Linear
        │
        ▼
    projections z (128-dim)  ← this is where the contrastive loss is applied

After pretraining, the projection head is DISCARDED.
Only the encoder + features h are used for linear evaluation / transfer learning.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ProjectionHead(nn.Module):
    """2-layer MLP projection head from the SimCLR paper.

    Maps encoder features into the space where contrastive loss is applied.
    Discarded after pretraining.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """SimCLR: encoder + projection head.

    The encoder is a ResNet-18 adapted for CIFAR-10 (32x32 images):
    - First conv: 3x3 stride 1 (instead of 7x7 stride 2) — preserves resolution
    - Max pooling removed — otherwise features collapse to 1x1 too early
    - Final FC layer replaced with Identity — we want raw features, not logits

    Returns BOTH features (h) and projections (z):
    - h: used for linear evaluation and any downstream task
    - z: used only during pretraining for the contrastive loss
    """

    def __init__(
        self,
        projection_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # ResNet-18 backbone
        backbone = resnet18(weights=None)  # no ImageNet pretraining — we train from scratch

        # Adapt first conv for 32x32 input (CIFAR-10 style)
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # Remove initial max pool (kills spatial info too early for 32x32)
        backbone.maxpool = nn.Identity()

        # Store feature dim BEFORE replacing the final layer
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet-18

        # Replace classification head with identity — we want raw features
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.projection_head = ProjectionHead(
            in_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of images, shape (B, 3, H, W).

        Returns:
            h: encoder features, shape (B, 512). Used for downstream tasks.
            z: projections, shape (B, projection_dim). Used for contrastive loss.
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only encoder features (for linear eval / inference)."""
        return self.encoder(x)


def build_simclr_model(config: dict) -> SimCLRModel:
    """Build a SimCLR model from a config dict."""
    return SimCLRModel(
        projection_dim=config["model"]["projection_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    )