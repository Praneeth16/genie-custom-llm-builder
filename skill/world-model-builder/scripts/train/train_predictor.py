"""Train the action-conditioned latent predictor on PushT, then test it three ways.

1. Rollout error: predicted vs true latents 1, 2, 4 and 8 steps ahead, against a baseline that
   assumes nothing changes.
2. Decoded position: the frozen position probe applied to the imagined latents, so the error
   reads in arena pixels (needs --probe).
3. Planning: CEM searches for the actions that reach the frame `--plan-horizon` steps ahead; the
   plan is compared with what the demonstrator did. Baselines: random actions, the mean action,
   and holding the current position (read from the current frame by the probe). With --probe
   the search also starts from the current position.

It also reports how much the imagined position degrades when the actions are shuffled between
examples: if that changes nothing, the predictor is ignoring the actions.

Test needs --score-test.

    python train/train_predictor.py --data runs/pusht --output runs/wm --max-steps 30   # probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jepa_wm.data import pusht
from jepa_wm.planning import cem
from jepa_wm.predictor import LatentPredictor
from jepa_wm.probe import AttentiveProbe
from jepa_wm.tracking import child_run

HORIZONS = (1, 2, 4, 8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder written by embed.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe", help="probe.pt from train_probe.py, to decode imagined latents")
    parser.add_argument("--score-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-horizon", type=int, default=3, help="steps rolled out in the training loss")
    parser.add_argument("--context", type=int, default=2, help="past frames the predictor sees")
    parser.add_argument("--plan-horizon", type=int, default=4)
    parser.add_argument("--plan-starts", type=int, default=200)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true", help="load predictor.pt from --output and only evaluate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    latents = torch.from_numpy(np.load(Path(args.data) / "latents.npy")).to(device)
    meta = np.load(Path(args.data) / "meta.npz")
    episode = meta["episode"]
    state = torch.from_numpy(meta["state"]).to(device)
    action = torch.from_numpy(meta["action"]).to(device)
    masks = {s: pusht.split_mask(episode, s) for s in pusht.SPLITS}
    splits = ["dev", "test"] if args.score_test else ["dev"]

    predictor = LatentPredictor(latents.shape[-1], latents.shape[1], action.shape[1], args.context).to(device)
    context = predictor.context
    sample = latents[torch.from_numpy(np.flatnonzero(masks["train"])[::4]).to(device)].float()
    predictor.set_normalization(sample.mean((0, 1)), sample.std((0, 1)))
    del sample

    def window(starts: torch.Tensor, horizon: int):
        """For frames t in `starts`: context latents, the next `horizon` actions and latents."""
        past = starts.unsqueeze(1) + torch.arange(1 - context, 1, device=device)
        future = starts.unsqueeze(1) + torch.arange(1, horizon + 1, device=device)
        return (
            predictor.normalize(latents[past].float()),
            action[future - 1],
            predictor.normalize(latents[future].float()),
        )

    def starts_for(split: str, horizon: int) -> torch.Tensor:
        return torch.from_numpy(pusht.sequence_starts(episode, masks[split], context - 1, horizon)).to(device)

    results = {"args": vars(args), "frames": {s: int(m.sum()) for s, m in masks.items()}}
    try:
        with child_run("latent_predictor", phase="predictor", seed=str(args.seed)) as log:
            optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.max_steps)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            train_starts = starts_for("train", args.train_horizon)
            step_seconds = []
            predictor.train()
            for step in range(0 if args.eval_only else args.max_steps):
                started = time.perf_counter()
                pick = torch.randint(len(train_starts), (args.batch_size,), generator=generator, device=device)
                past, actions, future = window(train_starts[pick], args.train_horizon)
                loss = F.mse_loss(predictor.rollout(past, actions), future)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                step_seconds.append(time.perf_counter() - started)
                if step % 100 == 0:
                    log({"predictor_loss": loss.item()}, step)
            if args.eval_only:
                predictor.load_state_dict(torch.load(output / "predictor.pt", map_location=device))
            else:
                torch.save(predictor.state_dict(), output / "predictor.pt")
            predictor.eval()
            report = {"seconds_per_step": float(np.median(step_seconds))} if step_seconds else {}
            if device.type == "cuda":
                report["peak_gpu_memory_fraction"] = (
                    torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                )
            results["training"] = report

            probe = None
            if args.probe:
                probe = AttentiveProbe(latents.shape[-1], 2).to(device)
                probe.load_state_dict(torch.load(args.probe, map_location=device))
                probe.eval()

            with torch.no_grad():
                for split in splits:
                    # 1 and 2: rollout error and decoded position, by horizon
                    starts = starts_for(split, max(HORIZONS))
                    sums = {}
                    # For the wrong-action control, pair every start with one half a split away.
                    partners = starts.roll(len(starts) // 2)
                    for chunk, partner in zip(starts.split(256), partners.split(256), strict=True):
                        past, actions, future = window(chunk, max(HORIZONS))
                        wrong_actions = window(partner, max(HORIZONS))[1]
                        imagined = predictor.rollout(past, actions)
                        unchanged = past[:, -1:].expand_as(future)
                        for h in HORIZONS:
                            rows = {
                                f"h{h}_latent_mse_model": F.mse_loss(imagined[:, h - 1], future[:, h - 1]),
                                f"h{h}_latent_mse_no_change": F.mse_loss(unchanged[:, h - 1], future[:, h - 1]),
                            }
                            truth = state[chunk + h]
                            # No model at all: an action is a target position, so guess that the
                            # agent is where the last action told it to go.
                            target_error = (action[chunk + h - 1] - truth).norm(dim=1).mean()
                            rows[f"h{h}_position_error_px_action_target"] = target_error * pusht.POSITION_SCALE
                            if probe:
                                shuffled = predictor.rollout(past, wrong_actions)
                                for name, guess in (
                                    ("model", imagined),
                                    ("no_change", unchanged),
                                    ("shuffled_actions", shuffled),
                                ):
                                    decoded = probe(predictor.denormalize(guess[:, h - 1]))
                                    rows[f"h{h}_position_error_px_{name}"] = (decoded - truth).norm(
                                        dim=1
                                    ).mean() * pusht.POSITION_SCALE
                            for key, value in rows.items():
                                sums[key] = sums.get(key, 0.0) + value.item() * len(chunk)
                    results[f"{split}_rollout"] = {k: v / len(starts) for k, v in sums.items()}

                    # 3: planning against the demonstrator's actions
                    starts = starts_for(split, args.plan_horizon)
                    picks = np.random.default_rng(args.seed).permutation(len(starts))[: args.plan_starts]
                    random = np.random.default_rng(args.seed + 1)
                    mean_action = action[torch.from_numpy(np.flatnonzero(masks["train"])).to(device)].mean(0)
                    errors = {"cem": [], "random": [], "mean_action": [], "hold_position": []}
                    # Many action paths reach the same goal, so also ask where each plan ends up:
                    # the last action is a target position; compare it with where the agent was
                    # at the goal frame.
                    ends = {name: [] for name in errors}
                    for t in starts[torch.from_numpy(picks).to(device)]:
                        past, actions, future = window(t.unsqueeze(0), args.plan_horizon)
                        here = probe(predictor.denormalize(past[:, -1]))[0].clamp(0, 1) if probe else None
                        plan = cem(
                            predictor,
                            past[0],
                            future[0, -1],
                            args.plan_horizon,
                            action.shape[1],
                            samples=args.cem_samples,
                            iterations=args.cem_iterations,
                            start_action=here,
                            seed=int(t),
                        )
                        guesses = {
                            "cem": plan,
                            "random": torch.from_numpy(random.uniform(size=plan.shape)).float().to(device),
                            "mean_action": mean_action.expand_as(plan),
                        }
                        if probe:
                            guesses["hold_position"] = here.expand_as(plan)
                        for name, guess in guesses.items():
                            errors[name].append((guess - actions[0]).norm(dim=1).mean().item())
                            ends[name].append((guess[-1] - state[t + args.plan_horizon]).norm().item())
                    results[f"{split}_planning"] = {
                        f"action_error_px_{k}": float(np.mean(v)) * pusht.POSITION_SCALE for k, v in errors.items() if v
                    } | {
                        f"plan_end_to_goal_px_{k}": float(np.mean(v)) * pusht.POSITION_SCALE
                        for k, v in ends.items()
                        if v
                    }
                    log({f"{split}_{k}": v for k, v in results[f"{split}_rollout"].items()})
                    log({f"{split}_{k}": v for k, v in results[f"{split}_planning"].items()})
    finally:
        (output / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
