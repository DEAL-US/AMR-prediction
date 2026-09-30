"""
Neural network architectures for per-antibiotic resistance prediction.

- BaselineMLP:   single-tower feed-forward network (for single feature source)
- TowerMLP:      encoder branch used inside the two-tower model
- TwoTowerMLP:   two parallel encoding branches + fusion head (for combined datasets)
"""
from typing import List

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Simple feed-forward MLP for binary classification.

    Architecture: [Linear -> ReLU -> Dropout] x N  ->  Linear(last_dim, 1)
    Output: raw logits (use BCEWithLogitsLoss for training).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TowerMLP(nn.Module):
    """
    Encoder branch that produces a fixed-size embedding.

    Same layer pattern as BaselineMLP but **without** the final classification head.
    Used as a building block for TwoTowerMLP.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) if len(self.net) > 0 else x


class TwoTowerMLP(nn.Module):
    """
    Two-tower model for datasets that combine two separate feature sources
    (e.g. NDARO gene features ``g_*`` and BAKTA UniRef features ``U_*``).

    Each tower independently encodes its feature block into an embedding.
    The two embeddings are concatenated and passed through a single linear
    fusion head to produce a binary classification logit.
    """

    def __init__(self, g_input_dim: int, u_input_dim: int,
                 hidden_dims: List[int], dropout: float):
        super().__init__()
        self.g_tower = TowerMLP(g_input_dim, hidden_dims, dropout)
        self.u_tower = TowerMLP(u_input_dim, hidden_dims, dropout)

        out_dim = hidden_dims[-1] if len(hidden_dims) else (g_input_dim + u_input_dim)
        fusion_in = out_dim * 2 if len(hidden_dims) else (g_input_dim + u_input_dim)
        self.fuse = nn.Sequential(nn.Linear(fusion_in, 1))

    def forward(self, x_g: torch.Tensor, x_u: torch.Tensor) -> torch.Tensor:
        z_g = self.g_tower(x_g)
        z_u = self.u_tower(x_u)
        z = torch.cat([z_g, z_u], dim=-1)
        return self.fuse(z).squeeze(-1)
