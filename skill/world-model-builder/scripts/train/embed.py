"""Encode every PushT frame with the frozen V-JEPA 2 encoder and cache the results.

Writes to --output:
  latents.npy          (N, grid*grid, dim) float16 token latents, for the probe and predictor
  meta.npz             state, action (N, 2) in [0, 1] and episode (N,)
  embeddings.parquet   one mean-pooled embedding per frame, for the Delta table and Vector Search

    python train/embed.py --output runs/pusht --limit 64   # probe
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jepa_wm.data import pusht
from jepa_wm.encoders import DEFAULT_REPO, FrameEncoder
from jepa_wm.tracking import child_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, help="only encode the first N frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch, child_run("embed_pusht", phase="embed", encoder=args.repo) as log:
        data = pusht.load(Path(scratch) / "pusht")
        data = {k: v[: args.limit] for k, v in data.items()}
        encoder = FrameEncoder(args.repo, args.grid).to(device)

        frames = torch.from_numpy(data["frames"])
        latents = np.empty((len(frames), args.grid**2, encoder.dim), dtype=np.float16)
        started = time.perf_counter()
        for i in range(0, len(frames), args.batch_size):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                latents[i : i + args.batch_size] = encoder(frames[i : i + args.batch_size].to(device)).float().cpu()
            if i % (args.batch_size * 20) == 0:
                log({"frames_encoded": i + args.batch_size}, step=i)
        seconds = time.perf_counter() - started
        report = {"frames": len(frames), "frames_per_second": len(frames) / seconds}
        if device.type == "cuda":
            report["peak_gpu_memory_fraction"] = (
                torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            )
        log(report)

        frame_in_episode = np.arange(len(frames)) - np.searchsorted(data["episode"], data["episode"])
        table = pd.DataFrame(
            {
                "frame_id": np.arange(len(frames)),
                "episode": data["episode"],
                "frame_in_episode": frame_in_episode,
                "split": np.select(
                    [pusht.split_mask(data["episode"], s) for s in pusht.SPLITS], list(pusht.SPLITS), default=""
                ),
                "agent_x": data["state"][:, 0],
                "agent_y": data["state"][:, 1],
                "embedding": list(latents.astype(np.float32).mean(1)),
            }
        )

        # UC Volumes do not support the random writes np.save and parquet writers can make,
        # so write locally and copy.
        np.save(Path(scratch) / "latents.npy", latents)
        np.savez(Path(scratch) / "meta.npz", state=data["state"], action=data["action"], episode=data["episode"])
        table.to_parquet(Path(scratch) / "embeddings.parquet", index=False)
        for name in ("latents.npy", "meta.npz", "embeddings.parquet"):
            shutil.copy(Path(scratch) / name, output / name)


if __name__ == "__main__":
    main()
