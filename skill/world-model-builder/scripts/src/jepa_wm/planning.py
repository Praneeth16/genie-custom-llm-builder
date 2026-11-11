"""Planning with a latent world model: search for actions whose imagined outcome matches a goal."""

from __future__ import annotations

import torch


@torch.no_grad()
def cem(
    predictor,
    context: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    action_dim: int,
    samples: int = 256,
    elites: int = 32,
    iterations: int = 5,
    start_action: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Cross-entropy method over action sequences in [0, 1].

    `context` is (context, L, dim) normalized latents of the frames seen so far and `goal` is
    (L, dim), the normalized latents of the frame to reach after `horizon` steps. Each round
    samples action sequences, imagines where they lead with `predictor.rollout`, keeps the
    closest `elites` and refits the sampling distribution to them. Returns (horizon, action_dim).

    `start_action` (action_dim,) centres the first round on a known sensible action, such as
    "stay where you are", with a narrower spread; without it the search starts from 0.5.
    """
    generator = torch.Generator().manual_seed(seed)
    if start_action is None:
        mean = torch.full((horizon, action_dim), 0.5, device=goal.device)
        std = torch.full((horizon, action_dim), 0.3, device=goal.device)
    else:
        mean = start_action.expand(horizon, action_dim).clone()
        std = torch.full((horizon, action_dim), 0.1, device=goal.device)
    start = context.unsqueeze(0).expand(samples, *context.shape)
    for _ in range(iterations):
        noise = torch.randn(samples, horizon, action_dim, generator=generator).to(goal.device)
        actions = (mean + std * noise).clamp(0, 1)
        reached = predictor.rollout(start, actions)[:, -1]
        cost = (reached - goal).pow(2).flatten(1).mean(1)
        best = actions[cost.topk(elites, largest=False).indices]
        mean, std = best.mean(0), best.std(0) + 1e-3
    return mean
