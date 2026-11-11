"""SIGReg: Sketched Isotropic Gaussian Regularization.

Written from the maths in Balestriero & LeCun, "LeJEPA" (arXiv 2511.08544), Definition 2:
project embeddings onto random unit directions and penalise each 1-D projection's distance
from a standard Gaussian with the Epps-Pulley characteristic-function statistic.

The paper's text weights the integrand with exp(-t^2 / sigma^2) while its code listing uses
exp(-t^2 / 2). This module uses exp(-t^2 / 2).
"""

from __future__ import annotations

import torch


def sigreg(
    z: torch.Tensor,
    num_directions: int = 256,
    seed: int | None = None,
    knots: int = 17,
    t_max: float = 5.0,
) -> torch.Tensor:
    """Mean Epps-Pulley statistic of `z` (N, D) over random unit directions.

    About 1 when `z` is isotropic standard Gaussian (the statistic is scaled by N, so its
    expectation there is sqrt(2*pi) - sqrt(2*pi/3), whatever N is); far larger when `z` collapses.
    Pass the training step as `seed` so every device draws the same directions.
    """
    n, d = z.shape
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    directions = torch.randn(d, num_directions, generator=generator).to(z)
    directions = directions / directions.norm(dim=0, keepdim=True)

    t = torch.linspace(-t_max, t_max, knots, device=z.device, dtype=z.dtype)
    x = (z @ directions).unsqueeze(-1) * t  # (N, M, K)
    gaussian = torch.exp(-0.5 * t**2)
    # |empirical CF - Gaussian CF|^2, split into real and imaginary parts
    error = (x.cos().mean(0) - gaussian) ** 2 + x.sin().mean(0) ** 2
    statistic = n * torch.trapezoid(error * gaussian, t, dim=-1)  # (M,)
    return statistic.mean()
