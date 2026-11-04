# How I Genie submission: Genie Code as your AI engineer

**Tier:** 2 (multi-step build plus a reusable asset)
**Surface:** Genie Code CLI 0.3.0 (`genie exec`, `/goal` via `create_goal`, custom skill), Databricks AI Runtime (H100), Unity Catalog, MLflow, FMAPI.
**Workspace:** `fe-vm-lakebase-praneeth`

## Summary (for the form)

I built a Genie Code skill, `custom-llm-builder`, that turns Genie into an AI engineer for small domain models. Given a task with a checkable metric, the skill makes Genie measure baselines first, compute signals that decide the training method (prompting, SFT, distillation, or RL), run the training on AI Runtime with TRL and Unsloth, compare methods at matched compute, and ship the winner to Unity Catalog with a batch scoring job.

Genie then used the skill end to end on a real task: extracting adverse drug events from medical case reports (PHEE V2, MIT license). A 4B open model (Qwen3-4B-Instruct-2507) fine-tuned with LoRA reached 0.782 dev strict event F1, against 0.716 for Claude Sonnet 5.5 with 5 examples and 0.301 for the base model. It runs about 30 times faster. A matched-compute experiment showed GRPO on top of SFT gives no extra gain on this task (0.7818 vs 0.7817), while extra SFT overfits (0.766), so RL is the better use of spare compute but not needed. On the held-out test split, scored once after the final choice was frozen, the fine-tuned model scores 0.786 strict event F1 with 99.7% valid outputs, against 0.708 for Claude Sonnet 5.5 with 5 examples, 0.635 for Claude zero-shot and 0.304 for the base model. It runs at 79 sentences per second on one H100 against 2.8 for Claude.

Along the way the build found and fixed problems that would have produced wrong conclusions: a schema that silently erased 18% of gold events, two scoring bugs, and an Unsloth chat-template bug that made sampled outputs unparsable and would have made RL look useless. Each fix went back into the skill as a rule or a guard.

## Links

- Repo (local, not yet published): `02-assets/genie-custom-llm-builder/`
- Skill: `skill/custom-llm-builder/` (installed at `~/.genie/skills/custom-llm-builder`)
- Strategy log with every decision and table: `examples/ade-phee/strategy_decision.md`
- Model card: `examples/ade-phee/MODEL_CARD.md`
- MLflow experiments: [baselines](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329213), [SFT](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329233), [RL](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/2384154102591201)
- UC model: `serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor` version 1, alias `champion`
- Batch scoring job (weekly, paused): [job 781370433162163](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/jobs/781370433162163?w=7474654904882204), output table `serverless_lakebase_praneeth_catalog.custom_llm.ade_events`
- Final AIR run (test, register, batch): [563423768528196](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/jobs/runs/563423768528196)
- Bugs and feedback: `sessions/BUGS.md` (7 items)

## Results

| Test split, 968 sentences | Strict event F1 | Valid JSON | PHEE EM | PHEE token F1 | Sentences/s |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 5.5, zero-shot | 0.635 | 94.7% | 0.480 | 0.639 | 2.7 |
| Claude Sonnet 5.5, 5-shot | 0.708 | 98.6% | 0.564 | 0.717 | 2.8 |
| Qwen3-4B base, 5-shot | 0.304 | 58.3% | 0.227 | 0.310 | 186 |
| **Qwen3-4B LoRA SFT (final)** | **0.786** | **99.7%** | **0.670** | **0.793** | **79** |
| Qwen3-4B SFT + GRPO (secondary) | 0.784 | 99.7% | 0.668 | 0.791 | 82 |

Dev matched-compute experiment, both arms 40 H100 minutes from the same SFT model (0.7817): continued SFT 0.7662, GRPO 0.7818.

## Genie Code CLI session IDs

| Session | What Genie did | Thread ID |
|---|---|---|
| A | Wrote the skill with `skill-creator`, then fixed 15 review findings | `01a1a310-330d-7cf3-9746-33e5f403c213` |
| B | Task card, PHEE V2 data prep, UC schema and volume, schema fix | `01a1a319-b67b-7952-a73e-c54520a1ec1c` |
| C | Frontier and base baselines, RL signals, first strategy decision | `01a1a323-262d-7bf3-ad9b-ad66ef1c7f48` |
| D | SFT probe and learning curve (500, 2,000, full) on H100 | `01a1a349-291a-73f2-867b-cf24252e1060` |
| E | `/goal`: sampling diagnosis, clean-template retrain, matched-compute GRPO vs SFT | `01a1a39f-f5ad-7d92-bd5c-237f85a6b6a3` |
| F | Test scoring, UC registration, batch job, model card | `01a1a54c-c5f1-7fd3-baaa-97bbaa35eb34` |

Raw JSONL event logs for every session are in `sessions/*.jsonl`, and the prompts given to Genie are in `sessions/*.prompt.md`. CLI sessions are stored locally only (see bug 4 in `sessions/BUGS.md`).

## Claude Code vs Genie Code

Claude Code planned the work, did the research (datasets, TRL and Unsloth versions, post-training methods), wrote each session brief, and reviewed every output. Genie Code did the building: it wrote the skill, prepared data, created UC objects, wrote and submitted every AI Runtime workload, trained and evaluated the models, and kept the decision log.

What worked well in Genie: deep Databricks knowledge (AIR YAML, UC, MLflow) out of the box, `--approve-for-me` refusing a fourth paid GPU run on its own, and `create_goal` keeping a long multi-run task on track.

Where review mattered: the schema that dropped gold events, the inverted image lesson, the GRPO batch shape, and the tie-break on the final model. These are listed with evidence in `sessions/BUGS.md` and `strategy_decision.md`.

## Reproduce

1. `ln -s $PWD/skill/custom-llm-builder ~/.genie/skills/custom-llm-builder`
2. `cd skill/custom-llm-builder && uv run --with pytest pytest tests -q`
3. In Genie Code CLI: `Use $custom-llm-builder to build a model for <task> with data <path>` and follow the phases.
