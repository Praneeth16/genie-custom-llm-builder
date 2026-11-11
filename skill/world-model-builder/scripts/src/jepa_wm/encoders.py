"""Frozen V-JEPA 2 encoder applied to single frames.

Weights are downloaded from the Hugging Face Hub at run time and are never stored in this
repo. Each frame is repeated to fill one tubelet (two frames), as V-JEPA 2's own
action-conditioned model does, so every frame gets its own latent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_REPO = "facebook/vjepa2-vitl-fpc64-256"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    def __init__(self, repo: str = DEFAULT_REPO, grid: int = 8, size: int = 256):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo).eval().requires_grad_(False)
        self.grid = grid
        self.size = size
        self.dim = self.model.config.hidden_size
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 frames (B, H, W, 3) -> latents (B, grid * grid, dim)."""
        x = frames.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.pixel_mean) / self.pixel_std
        video = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)  # (B, T=2, C, H, W): one tubelet
        tokens = self.model(pixel_values_videos=video, skip_predictor=True).last_hidden_state
        side = int(tokens.shape[1] ** 0.5)
        tokens = tokens.transpose(1, 2).reshape(len(frames), self.dim, side, side)
        return F.adaptive_avg_pool2d(tokens, self.grid).flatten(2).transpose(1, 2)
