import numpy as np
import torch

from jepa_wm.data.pusht import sequence_starts, split_mask
from jepa_wm.planning import cem
from jepa_wm.predictor import LatentPredictor
from jepa_wm.probe import AttentiveProbe


def test_splits_partition_episodes():
    episode = np.repeat(np.arange(20), 5)
    masks = [split_mask(episode, s) for s in ("train", "dev", "test")]
    assert np.sum(masks, axis=0).tolist() == [1] * 100
    assert [int(m.sum()) for m in masks] == [80, 10, 10]


def test_sequence_starts_never_cross_an_episode_boundary():
    episode = np.repeat(np.arange(4), 6)
    starts = sequence_starts(episode, np.ones(24, dtype=bool), before=1, after=2)
    assert starts.tolist() == [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]


def test_untrained_predictor_predicts_no_change():
    predictor = LatentPredictor(dim=16, n_tokens=4, action_dim=2, width=32, depth=1, heads=2)
    latents = torch.randn(3, 2, 4, 16)
    assert torch.equal(predictor(latents, torch.rand(3, 2)), latents[:, -1])
    assert predictor.rollout(latents, torch.rand(3, 5, 2)).shape == (3, 5, 4, 16)


def test_probe_output_shape():
    assert AttentiveProbe(dim=16, out_dim=2, width=32, heads=2)(torch.randn(5, 4, 16)).shape == (5, 2)


class MovesToAction:
    """Toy world: the single latent token is the position, and it jumps to the action."""

    def rollout(self, latents, actions):
        return actions.unsqueeze(2)


def test_cem_finds_the_action_that_reaches_the_goal():
    goal = torch.tensor([[0.2, 0.9]])
    plan = cem(MovesToAction(), torch.zeros(2, 1, 2), goal, horizon=3, action_dim=2, iterations=8)
    assert torch.allclose(plan[-1], goal[0], atol=0.03)
