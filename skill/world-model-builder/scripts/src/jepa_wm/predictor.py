"""Action-conditioned latent predictor: the world-model half of a JEPA.

Given the latents of the last few frames and the action taken, predict the next frame's
latents. Rolling it forward imagines a future without rendering a single pixel.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        n_tokens: int,
        action_dim: int,
        context: int = 2,
        width: int = 384,
        depth: int = 4,
        heads: int = 6,
    ):
        super().__init__()
        self.context = context
        self.n_tokens = n_tokens
        self.embed = nn.Linear(dim, width)
        self.action = nn.Linear(action_dim, width)
        self.pos = nn.Parameter(torch.randn(context, n_tokens, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)  # start as "nothing changes"; training learns the change
        nn.init.zeros_(self.out.bias)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean) / self.std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.std + self.mean

    def forward(self, latents: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Normalized latents (B, context, L, dim) and action (B, A) -> next latents (B, L, dim)."""
        # The action is added to every token. As a single extra token it was easy to ignore.
        h = self.embed(latents) + self.pos + self.action(action)[:, None, None, :]
        h = self.blocks(h.flatten(1, 2))
        return latents[:, -1] + self.out(h[:, -self.n_tokens :])

    def rollout(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply actions (B, H, A) one after another -> predicted latents (B, H, L, dim)."""
        steps = []
        for h in range(actions.shape[1]):
            steps.append(self(latents, actions[:, h]))
            latents = torch.cat([latents[:, 1:], steps[-1].unsqueeze(1)], dim=1)
        return torch.stack(steps, dim=1)
