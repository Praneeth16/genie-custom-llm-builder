"""Train the sensor JEPA and its baselines on synthetic telemetry, then score them.

Baselines: the same network reconstructing raw values, a Gaussian fitted to raw patches with
no network at all, and an isolation forest.

Trains on anomaly-free data and reports each anomaly family separately. Dev is for choosing
settings and the F1 threshold. Test is only scored with `--score-test`, once the settings are
frozen. No checkpoint selection: the last step is the one kept.

    python train/train_sensor_jepa.py --output runs/sensor --max-steps 30   # probe
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from jepa_wm.data.synthetic_sensor import ANOMALY_KINDS, generate, windows
from jepa_wm.sensor_jepa import SensorMaskedModel
from jepa_wm.tracking import child_run

METHODS = {"jepa": "latent", "masked_reconstruction": "input", "gaussian": None, "isolation_forest": None}


def evaluate(scorer, splits: dict, score_test: bool, prefix: str = "") -> dict:
    """`scorer(x) -> scores`; `splits[family] = {"dev": (x, y), "test": (x, y)}`."""
    report = {}
    for family, split in splits.items():
        family = prefix + family
        dev_scores, dev_labels = scorer(split["dev"][0]), split["dev"][1]
        report[f"{family}_dev_auroc"] = roc_auc_score(dev_labels, dev_scores)
        report[f"{family}_dev_auprc"] = average_precision_score(dev_labels, dev_scores)
        if score_test:
            precision, recall, thresholds = precision_recall_curve(dev_labels, dev_scores)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            threshold = thresholds[min(int(f1.argmax()), len(thresholds) - 1)]
            test_scores, test_labels = scorer(split["test"][0]), split["test"][1]
            predicted = test_scores >= threshold
            report[f"{family}_test_auroc"] = roc_auc_score(test_labels, test_scores)
            report[f"{family}_test_auprc"] = average_precision_score(test_labels, test_scores)
            report[f"{family}_test_f1"] = 2 * (predicted & test_labels).sum() / (predicted.sum() + test_labels.sum())
    return report


def gaussian_scorer(train_x: torch.Tensor, patch: int = 8):
    """No network: Mahalanobis distance of each raw patch from the normal data, worst patch wins."""

    def patches(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(len(x), x.shape[1] // patch, -1)

    normal = patches(train_x).flatten(0, 1)
    mean = normal.mean(0)
    precision = torch.linalg.inv(torch.cov(normal.T) + 1e-4 * torch.eye(normal.shape[1]))

    def scores(x: np.ndarray) -> np.ndarray:
        offset = patches(torch.from_numpy(x)) - mean
        return ((offset @ precision) * offset).sum(-1).amax(1).numpy()

    return scores


def forest_scores(forest, x: np.ndarray) -> np.ndarray:
    return -forest.score_samples(x.reshape(len(x), -1))


@torch.no_grad()
def model_scores(model, device, method: str, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    x = torch.from_numpy(x)
    return torch.cat([model.score(x[i : i + batch].to(device), method) for i in range(0, len(x), batch)]).cpu().numpy()


def train(model, x: torch.Tensor, args, device, log) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.max_steps, pct_start=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    n_tokens = model.encoder.n_tokens
    block = n_tokens // 4
    step_seconds = []
    for step in range(args.max_steps):
        started = time.perf_counter()
        batch = x[torch.randint(len(x), (args.batch_size,), generator=generator)].to(device)
        start = int(torch.randint(n_tokens - block + 1, (1,), generator=generator))
        masked = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        masked[start : start + block] = True
        loss, parts = model.loss(batch, masked, lam=args.lam, seed=step)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}: {parts}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        step_seconds.append(time.perf_counter() - started)
        if step % 100 == 0 or step == args.max_steps - 1:
            log({f"{model.target}_{k}": v for k, v in parts.items()}, step)
    report = {"seconds_per_step": float(np.median(step_seconds)), **{f"final_{k}": v for k, v in parts.items()}}
    if device.type == "cuda":
        report["peak_gpu_memory_fraction"] = (
            torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--score-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.05, help="SIGReg weight in the JEPA loss")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=400_000, help="length of the training series")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_x = torch.from_numpy(windows(*generate(args.train_steps, args.channels, seed=args.seed), args.window)[0])
    mean, std = train_x.mean(dim=(0, 1)), train_x.std(dim=(0, 1))
    splits = {
        family: {
            name: windows(*generate(100_000, args.channels, 0.05, family, seed=args.seed + offset), args.window)
            for name, offset in (("dev", 1), ("test", 2))
        }
        for family in ANOMALY_KINDS
    }

    results = {"args": vars(args), "device": str(device), "train_windows": len(train_x)}
    try:
        tags = {"phase": "train", "method": "sensor_jepa_sigreg", "seed": str(args.seed)}
        with child_run(f"sensor_seed_{args.seed}", **tags) as log:
            for name in args.methods.split(","):
                if name == "gaussian":
                    report = evaluate(gaussian_scorer(train_x[::4]), splits, args.score_test)
                elif name == "isolation_forest":
                    forest = IsolationForest(n_estimators=200, random_state=args.seed)
                    forest.fit(train_x[::8].flatten(1).numpy())
                    report = evaluate(partial(forest_scores, forest), splits, args.score_test)
                else:
                    model = SensorMaskedModel(args.channels, args.window, target=METHODS[name]).to(device)
                    model.set_normalization(mean.to(device), std.to(device))
                    report = train(model, train_x, args, device, log)
                    model.eval()
                    model.fit_density(train_x[::4].to(device))
                    for method in ("density", "prediction"):
                        scorer = partial(model_scores, model, device, method)
                        report |= evaluate(scorer, splits, args.score_test, prefix=f"{method}_")
                    torch.save(model.state_dict(), output / f"{name}.pt")
                results[name] = report
                log({f"{name}_{k}": float(v) for k, v in report.items()})
    finally:
        (output / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
