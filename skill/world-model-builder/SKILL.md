---
name: world-model-builder
description: Build and evaluate JEPA-style models on Databricks AI Runtime for video, robot, or sensor data. Covers a probe on a frozen V-JEPA 2 encoder, an action-conditioned latent predictor, and a small JEPA trained from scratch with SIGReg, each measured against a baseline that needs no model.
metadata:
  short-description: Build measured JEPA and world models
---

# World Model Builder

Build the cheapest model that meets the user's target, and prove it against a baseline that needs no neural network. Use `databricks-ai-runtime` for workload mechanics, `databricks-model-serving` for endpoints, and `databricks-vector-search` for indexes. Do not copy those skills into the project.

A JEPA predicts its own latents for a hidden or future part of the input. It does not reconstruct raw values. An encoder alone describes the present. A world model adds a predictor that takes the current latents and an action and returns the next latents.

## Workflow

Each phase ends at a measured gate. Record inputs, outputs, metrics, run IDs, and decisions.

1. **Task card.** Write `task_card.yaml` in the user's project. Capture the data and its license, the modality, whether actions are recorded, the frozen metric, the splits, and the latency and cost targets. Split by episode or by time, never by frame. Gate: the metric runs on hand-checked examples. Read [licensing](references/licensing.md).
2. **Baselines.** Measure at least one baseline that needs no trained network on the dev split, e.g., a Gaussian fitted to raw values. For a learned network, also score the same network with random weights. Gate: if a baseline meets every target, stop and say so. Read [evaluation protocol](references/eval-protocol.md).
3. **Level selection.** Choose the lowest level that can meet the target from the table in [level selection](references/level-selection.md). Log the level and the reason as MLflow tags. Load only the selected recipe.
4. **Train on AI Runtime.** Dry-run the workload, then run a 30-step probe, then the full run. Gate: memory, step time, and loss are sane in the probe, and the full run beats the baseline on dev. Read [AIR glue](references/air-glue.md).
5. **Evaluate and ship.** Freeze the settings, then score the test split once. Register the model in Unity Catalog with the `champion` alias. Ship batch scoring first, then an endpoint if the user needs one. Write a model card that includes every baseline. Gate: ship only what beat its baseline, and report the rest as measured.

## Hard rules

- Never tune or pick settings on the test split. The training scripts only score test with `--score-test`.
- Never report a learned model without the baseline next to it. In the reference build a Gaussian with no network beat the sensor JEPA at anomaly detection, and so did the same network with random weights.
- Give the baseline the same information the model gets. If the model sees actions, include a baseline that uses the actions without a model. In the reference build the guess "the agent is where the last action sent it" beat the latent predictor beyond one step.
- Check that a predictor uses its actions. Roll it out with actions taken from distant examples. If the error does not rise, the predictor is ignoring the actions.
- Do not claim that planning works from an offline test alone. In the reference build the planner did not beat "stay in place".
- Never copy code from the LeJEPA reference repository or from `jepa-wms`. Both are CC-BY-NC. Write SIGReg from the paper, as `scripts/src/jepa_wm/sigreg.py` does.
- Never commit model weights or datasets. Download them at run time.
- Use `environment.version: databricks_ai_v6` and `max_retries: 0`. Do not call `mlflow.set_experiment` inside a run.
- Never cancel an AI Runtime run you did not start.
- Send frames or windows to an endpoint, not whole videos. Model Serving limits a request to 16 MB.

`scripts/` is a copy of the reference repo `jepa-world-models-databricks` at the commit in `scripts/SOURCE_COMMIT`. `scripts/src/jepa_wm/` is the package, `scripts/train/` holds one script per step, `scripts/workloads/` holds the AI Runtime workload files, and `scripts/scripts/air.sh` submits a workload with the values in `scripts/.env`. File paths in `references/` are relative to `scripts/`, e.g., `train/embed.py` is `scripts/train/embed.py`. PushT and the synthetic sensor generator are examples, not restrictions.
