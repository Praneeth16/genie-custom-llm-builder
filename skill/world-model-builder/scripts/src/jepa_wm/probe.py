"""Attentive probe: a small head trained on frozen token latents."""

from __future__ import annotations

import torch
from torch import nn


class AttentiveProbe(nn.Module):
    """One learned query cross-attends over the tokens; a linear layer reads out the answer."""

    def __init__(self, dim: int, out_dim: int, width: int = 256, heads: int = 8):
        super().__init__()
        self.project = nn.Linear(dim, width)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Raw encoder latents (B, L, dim) -> (B, out_dim)."""
        h = self.project((tokens - self.mean) / self.std)
        pooled, _ = self.attention(self.query.expand(len(h), -1, -1), h, h)
        return self.head(self.norm(pooled.squeeze(1)))
