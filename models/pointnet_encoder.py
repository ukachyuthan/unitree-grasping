"""
PointNet encoder for grasping.

Architecture (Qi et al., 2017):
    input  (B, N, 3)
    ↓  shared MLP [3 → 64 → 128]        per-point features
    ↓  symmetric max-pool                (B, 128)
    ↓  MLP [128 → 256 → embed_dim]      global descriptor
    output (B, embed_dim)

Design choices for this project:
- No T-Net (input / feature transform).  T-Net helps accuracy but adds
  training complexity and is not needed for a fixed-view overhead camera.
  Add it later if the policy struggles to generalise across rotations.
- embed_dim = 128 by default — compact enough to concatenate with a 22-dim
  proprio vector and still run a shallow MLP policy head.
- All weights are trainable end-to-end via PPO policy gradient.
  The encoder is part of the actor and the critic in PointNetActorCritic,
  with independent weight copies (not shared) so actor and critic can
  specialise their representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    Encodes a variable-length point cloud to a fixed-size global descriptor.

    Args:
        embed_dim: Dimension of the output embedding.  Default 128.
        input_dim: Input feature dimension per point.  Default 3 (XYZ).
    """

    def __init__(self, embed_dim: int = 128, input_dim: int = 3):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-point MLP (shared weights = 1-D convolution over point axis)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Global MLP after max-pool
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pointcloud: (B, N, 3)  — N points in robot-base frame
        Returns:
            embedding:  (B, embed_dim)
        """
        # (B, 3, N) for Conv1d
        x = pointcloud.transpose(1, 2)
        x = self.mlp1(x)              # (B, 128, N)
        x = x.max(dim=-1).values      # (B, 128)  symmetric max-pool
        x = self.mlp2(x)              # (B, embed_dim)
        return x

    @staticmethod
    def from_flat(pc_flat: torch.Tensor, num_points: int) -> "PointNetEncoder":
        """Helper: reshape flat obs slice before passing to forward."""
        B = pc_flat.shape[0]
        return pc_flat.view(B, num_points, 3)


def build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    """Utility: build a plain MLP with the given layer sizes."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
