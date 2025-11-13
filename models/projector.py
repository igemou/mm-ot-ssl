import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLPProjector(nn.Module):
    """
    Simple 2-layer MLP projector to map modality-specific embeddings
    to a shared representation.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: Optional[int] = None,
        use_bn: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z
