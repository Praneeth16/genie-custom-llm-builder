"""Masked prediction models for multivariate sensor windows.

`target="latent"` is the JEPA: predict the encoder's own latents for a masked block of time,
with SIGReg keeping the latents from collapsing (no teacher network, no stop-gradient).
`target="input"` is the generative baseline: same network, but it reconstructs the masked
raw values. A window's anomaly score is either its worst masked-block prediction error or
how far its token latents sit from those of normal data.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from jepa_wm.sigreg import sigreg


def _transformer(dim: int, depth: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class SensorEncoder(nn.Module):
    def __init__(self, n_channels: int, window: int, patch: int, dim: int, depth: int, heads: int):
        super().__init__()
        self.patch = patch
        self.n_tokens = window // patch
        self.embed = nn.Linear(n_channels * patch, dim)
        self.pos = nn.Parameter(torch.randn(self.n_tokens, dim) * 0.02)
        self.blocks = _transformer(dim, depth, heads)

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, L, patch * C)"""
        return x.reshape(x.shape[0], self.n_tokens, -1)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embed(self.patches(x)) + self.pos
        if keep is not None:
            h = h[:, keep]
        # No final LayerNorm: it would pin every token to the same scale, which fights SIGReg's
        # Gaussian target and hid anomalies from the density score in dev experiments.
        return self.blocks(h)


class SensorMaskedModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        window: int = 128,
        patch: int = 8,
        dim: int = 128,
        depth: int = 4,
        predictor_depth: int = 2,
        heads: int = 4,
        target: str = "latent",
    ):
        super().__init__()
        if target not in ("latent", "input"):
            raise ValueError(f"target must be 'latent' or 'input', got {target!r}")
        self.target = target
        self.encoder = SensorEncoder(n_channels, window, patch, dim, depth, heads)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.predictor_pos = nn.Parameter(torch.randn(self.encoder.n_tokens, dim) * 0.02)
        self.predictor = _transformer(dim, predictor_depth, heads)
        self.head = nn.Linear(dim, dim if target == "latent" else n_channels * patch)
        self.register_buffer("mean", torch.zeros(n_channels))
        self.register_buffer("std", torch.ones(n_channels))
        self.register_buffer("density_mean", torch.zeros(dim))
        self.register_buffer("density_precision", torch.eye(dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Window embedding (B, dim): mean of the encoder tokens."""
        return self.encoder((x - self.mean) / self.std).mean(1)

    def _errors(self, x: torch.Tensor, masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Error per masked token (B, n_masked) and the full-window latents (B, L, dim)."""
        x = (x - self.mean) / self.std
        latents = self.encoder(x)
        context = self.encoder(x, ~masked)
        h = self.mask_token.expand(x.shape[0], self.encoder.n_tokens, -1).clone()
        h[:, ~masked] = context
        prediction = self.head(self.predictor(h + self.predictor_pos)[:, masked])
        wanted = latents[:, masked] if self.target == "latent" else self.encoder.patches(x)[:, masked]
        return F.mse_loss(prediction, wanted, reduction="none").mean(-1), latents

    def loss(self, x: torch.Tensor, masked: torch.Tensor, lam: float = 0.05, seed: int | None = None):
        errors, latents = self._errors(x, masked)
        prediction = errors.mean()
        if self.target == "input":
            return prediction, {"prediction": prediction.item()}
        regularizer = sigreg(latents.mean(1), seed=seed)
        total = lam * regularizer + (1 - lam) * prediction
        return total, {"prediction": prediction.item(), "sigreg": regularizer.item()}

    def block_masks(self, blocks: int = 4) -> torch.Tensor:
        """(blocks, L) boolean masks that tile the window with contiguous blocks."""
        n = self.encoder.n_tokens
        index = torch.arange(n, device=self.mean.device) * blocks // n
        return index.unsqueeze(0) == torch.arange(blocks, device=index.device).unsqueeze(1)

    @torch.no_grad()
    def fit_density(self, x: torch.Tensor, batch: int = 1024) -> None:
        """Fit a Gaussian to the token latents of normal windows, for `score(..., "density")`."""
        tokens = torch.cat([self.encoder((x[i : i + batch] - self.mean) / self.std) for i in range(0, len(x), batch)])
        tokens = tokens.flatten(0, 1)
        covariance = torch.cov(tokens.T) + 1e-4 * torch.eye(tokens.shape[1], device=tokens.device)
        self.density_mean.copy_(tokens.mean(0))
        self.density_precision.copy_(torch.linalg.inv(covariance))

    @torch.no_grad()
    def score(self, x: torch.Tensor, method: str = "density") -> torch.Tensor:
        """Anomaly score (B,).

        "density": the largest Mahalanobis distance of any token latent from the normal data.
        "prediction": the largest masked-block prediction error over the tiling masks.
        """
        if method == "prediction":
            return torch.stack([self._errors(x, m)[0].mean(1) for m in self.block_masks()]).amax(0)
        offset = self.encoder((x - self.mean) / self.std) - self.density_mean
        return ((offset @ self.density_precision) * offset).sum(-1).amax(1)
