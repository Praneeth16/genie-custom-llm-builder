"""Train an attentive probe that reads the agent's position from frozen V-JEPA 2 latents.

Compared with ridge regression on downsampled pixels and on the mean-pooled latent. Errors
are distances in arena pixels (the arena is 512 x 512). Test needs --score-test.

    python train/train_probe.py --data runs/pusht --output runs/probe --max-steps 30   # probe
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from jepa_wm.data import pusht
from jepa_wm.probe import AttentiveProbe
from jepa_wm.tracking import child_run


def pixel_error(predicted: np.ndarray, state: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - state, axis=1).mean() * pusht.POSITION_SCALE)


def ridge_errors(features: np.ndarray, state: np.ndarray, masks: dict, splits: list[str]) -> dict:
    """Fit on train, pick the penalty on dev, report the chosen model on `splits`."""
    fits = [Ridge(alpha=a).fit(features[masks["train"]], state[masks["train"]]) for a in (0.1, 1, 10, 100, 1000)]
    best = min(fits, key=lambda fit: pixel_error(fit.predict(features[masks["dev"]]), state[masks["dev"]]))
    return {f"{s}_error_px": pixel_error(best.predict(features[masks[s]]), state[masks[s]]) for s in splits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder written by embed.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    latents = torch.from_numpy(np.load(Path(args.data) / "latents.npy")).to(device)
    meta = np.load(Path(args.data) / "meta.npz")
    state = meta["state"]
    masks = {s: pusht.split_mask(meta["episode"], s) for s in pusht.SPLITS}
    splits = ["dev", "test"] if args.score_test else ["dev"]
    index = {s: torch.from_numpy(np.flatnonzero(m)).to(device) for s, m in masks.items()}
    target = torch.from_numpy(state).to(device)

    results = {"args": vars(args), "frames": {s: int(m.sum()) for s, m in masks.items()}}
    try:
        with child_run("probe_agent_position", phase="probe", seed=str(args.seed)) as log:
            probe = AttentiveProbe(latents.shape[-1], 2).to(device)
            train_latents = latents[index["train"]].float()
            probe.set_normalization(train_latents.mean((0, 1)), train_latents.std((0, 1)))
            del train_latents
            optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.max_steps)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            step_seconds = []
            for step in range(args.max_steps):
                started = time.perf_counter()
                pick = torch.randint(len(index["train"]), (args.batch_size,), generator=generator, device=device)
                batch = index["train"][pick]
                loss = F.mse_loss(probe(latents[batch].float()), target[batch])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                step_seconds.append(time.perf_counter() - started)
                if step % 100 == 0:
                    log({"probe_loss": loss.item()}, step)

            probe.eval()
            report = {"seconds_per_step": float(np.median(step_seconds))}
            with torch.no_grad():
                for s in splits:
                    chunks = [probe(latents[i].float()) for i in index[s].split(1024)]
                    report[f"{s}_error_px"] = pixel_error(torch.cat(chunks).cpu().numpy(), state[masks[s]])
            if device.type == "cuda":
                report["peak_gpu_memory_fraction"] = (
                    torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                )
            results["attentive_probe"] = report
            torch.save(probe.state_dict(), output / "probe.pt")

            pooled = latents.float().mean(1).cpu().numpy()
            results["ridge_pooled_latent"] = ridge_errors(pooled, state, masks, splits)

            with tempfile.TemporaryDirectory() as scratch:
                frames = torch.from_numpy(pusht.load(Path(scratch) / "pusht")["frames"][: len(state)])
            small = F.avg_pool2d(frames.permute(0, 3, 1, 2).float() / 255, 4).flatten(1).numpy()  # 24x24x3
            results["ridge_pixels"] = ridge_errors(small, state, masks, splits)

            mean_position = state[masks["train"]].mean(0)
            results["predict_mean"] = {
                f"{s}_error_px": pixel_error(np.tile(mean_position, (int(masks[s].sum()), 1)), state[masks[s]])
                for s in splits
            }
            for name in ("attentive_probe", "ridge_pooled_latent", "ridge_pixels", "predict_mean"):
                log({f"{name}_{k}": v for k, v in results[name].items()})
    finally:
        (output / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
