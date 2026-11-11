"""PushT from the Hugging Face Hub (`lerobot/pusht`, LeRobot v3 layout).

A 2-D pushing task: 206 episodes, 25,650 frames of 96x96 video at 10 fps, with the agent's
(x, y) position as state and the target (x, y) as action. Read directly from the parquet and
mp4 files, so the `lerobot` package is not needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = "lerobot/pusht"
FILES = ("data/chunk-000/file-000.parquet", "videos/observation.image/chunk-000/file-000.mp4")
# Episodes, not frames, are split, so no episode is seen in both training and evaluation.
# Fractions of the episodes present, in episode order.
SPLITS = {"train": (0.0, 0.8), "dev": (0.8, 0.9), "test": (0.9, 1.0)}
POSITION_SCALE = 512.0  # states and actions are pixel coordinates in a 512x512 arena


def load(cache_dir: str | Path) -> dict[str, np.ndarray]:
    """Return frames (N, 96, 96, 3) uint8, state and action (N, 2) in [0, 1], episode (N,)."""
    import av
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table_path, video_path = (hf_hub_download(REPO, f, repo_type="dataset", local_dir=cache_dir) for f in FILES)
    table = pq.read_table(table_path, columns=["observation.state", "action", "episode_index", "index"]).to_pandas()
    if not (table["index"].to_numpy() == np.arange(len(table))).all():
        raise ValueError("rows are not in frame order")

    with av.open(video_path) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    if len(frames) != len(table):
        raise ValueError(f"{len(frames)} video frames for {len(table)} rows")

    return {
        "frames": frames,
        "state": np.stack(table["observation.state"].to_numpy()).astype(np.float32) / POSITION_SCALE,
        "action": np.stack(table["action"].to_numpy()).astype(np.float32) / POSITION_SCALE,
        "episode": table["episode_index"].to_numpy(),
    }


def split_mask(episode: np.ndarray, split: str) -> np.ndarray:
    episodes = np.unique(episode)
    low, high = (int(fraction * len(episodes)) for fraction in SPLITS[split])
    return np.isin(episode, episodes[low:high])


def sequence_starts(episode: np.ndarray, mask: np.ndarray, before: int, after: int) -> np.ndarray:
    """Frame indices t where frames t-before .. t+after all lie in one episode of the split."""
    t = np.arange(before, len(episode) - after)
    return t[mask[t] & (episode[t - before] == episode[t + after])]
