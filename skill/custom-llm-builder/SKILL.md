---
name: custom-llm-builder
description: Build and evaluate a domain-specific small open LLM for verifiable extraction or classification tasks on Databricks AI Runtime, choosing prompt optimization, distillation, SFT, or RL from measured evidence.
metadata:
  short-description: Build measured task-specific small LLMs
---

# Custom LLM Builder

Build the smallest model that meets the user's quality, latency, and cost target. Use `databricks-ai-runtime` for workload mechanics and `databricks-mlflow-evaluation` for MLflow evaluation and GEPA. Do not copy those skills into the project.

## Workflow

Each phase ends at a measured gate. Record inputs, outputs, metrics, run IDs, and decisions.

1. **Task card.** Write `task_card.yaml` in the user's project. Capture the task, data source and license, output schema, frozen metric, latency target, and cost target. Gate: the schema and metric run on hand-checked examples. Read [strategy selection](references/strategy-selection.md) and [rewards](references/rewards.md).
2. **Baselines.** Measure frontier zero-shot and few-shot through FMAPI, base-small-model zero-shot, and optionally GEPA. Gate: if prompting meets every target, stop and say so. Read [GEPA baseline](references/recipes/gepa-baseline.md) and [evaluation protocol](references/eval-protocol.md).
3. **Strategy selection.** Measure gold-label availability, base valid-format rate, reward variance over N samples per prompt, teacher logprob availability, and train size. Select a method from the decision table. Log the method and reason as MLflow tags. Load only the selected recipe.
4. **Train on AI Runtime.** Run a 30-step probe first. Measure peak memory, step time, completion clipped ratio, and zero-standard-deviation group fraction. Gate: proceed only when memory, throughput, and learning-signal checks pass. For RL, verify tuned rollouts score differently from base rollouts on identical prompts. Read [AIR glue](references/air-glue.md).
5. **Evaluate and ship.** Score the frozen test split once. Compare every baseline and compare SFT with SFT+RL at matched compute. Keep or discard changes in a bounded `/goal` loop with an explicit run budget. Register only the best model in Unity Catalog, create a batch-scoring workload, and write a model card. Gate: ship only if the model meets the task card and the comparison is honest.

## Hard Rules

- Never tune, select prompts, or choose checkpoints on the test split.
- Never assume RL improves extraction. Test SFT+RL against SFT at matched compute.
- Never cancel an AI Runtime run you did not start.
- Pin `trl==1.15.0` with `unsloth==2026.11.1`. Do not upgrade TRL independently.
- Do not call `mlflow.set_experiment` inside AI Runtime runs.
- Use `max_retries: 0` while developing and dry-run workload YAML before submission.
- Preserve valid empty examples. Do not add a standalone format reward.
- Treat serving endpoints as optional. Ship batch scoring first.

Use `scripts/task_adapter.py` and `scripts/rewards.py` as the task contract. The included PHEE adapter is an example, not a restriction.
