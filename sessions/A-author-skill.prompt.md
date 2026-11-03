Use $skill-creator to author a new Genie skill named `custom-llm-builder` in `skill/custom-llm-builder/` of this repo.

## Goal
The skill turns Genie into an AI engineer that builds a better domain-specific small open LLM (about 4B parameters) for a user's task on Databricks AI Runtime. Its core job is to choose the right post-training strategy for the task from measured evidence (prompt optimization, LoRA SFT, off-policy distillation, on-policy distillation with GKD, GRPO-family RL, or SFT then RL), run it with TRL and Unsloth on AI Runtime, evaluate it honestly against baselines, and ship it to Unity Catalog. It must work for any extraction or classification task with a verifiable metric, not only the first example.

## Inputs (read all of them first)
- `research/01-strategy-selection.md`: decision table and evidence. Key point: RL on top of SFT is not proven for extraction, so the skill must test it against SFT at matched compute rather than assume it helps.
- `research/02-stack-trl-unsloth.md`: versions, pins, model choice, GPU options, TRL GRPO API, Unsloth gotchas.
- `research/03-dataset-phee.md`: the first worked example (adverse drug event extraction on PHEE) with the output schema and reward spec.
- `research/04-databricks-facts-and-gotchas.md`: workspace facts and AI Runtime gotchas verified in this workspace.
- The built-in skills `databricks-ai-runtime` and `databricks-mlflow-evaluation`. Delegate to them; do not copy their content.

## Required structure
```
skill/custom-llm-builder/
  SKILL.md
  agents/openai.yaml          # same format as the built-in Databricks skills
  references/strategy-selection.md
  references/recipes/sft-lora.md
  references/recipes/distill-offpolicy.md
  references/recipes/gkd-onpolicy.md
  references/recipes/grpo-dapo.md
  references/recipes/gepa-baseline.md
  references/rewards.md
  references/eval-protocol.md
  references/air-glue.md
  scripts/                    # see below
  tests/                      # pytest for scripts that can run on CPU
```

## SKILL.md workflow (five phases, each ends with a measured gate)
1. Task card: task, data source and license, output schema, metric, latency and cost targets. Write it to `task_card.yaml` in the user's project.
2. Baselines: frontier model zero-shot and few-shot through Databricks FMAPI, base small model zero-shot, optional GEPA. Gate: if prompting meets the target, stop and say so.
3. Strategy selection: compute the signals (gold labels present, base-model valid-format rate, reward variance over N samples per prompt, teacher logprobs available, train size) and pick a method using the decision table. Log the choice and the reason as MLflow tags.
4. Train on AI Runtime: always a 30-step probe first (peak memory, step time, clipped ratio, zero-std group fraction), then the full run. For RL, check that tuned rollouts score differently from base rollouts on the same prompts (guards the Unsloth LoRA-not-injected bug).
5. Evaluate and ship: score the frozen test split once, compare against every baseline, compare SFT vs SFT+RL at matched compute, keep or discard changes in a bounded loop (`/goal` with a run budget), then register the best model in Unity Catalog, create a batch scoring workload, and write a model card.

Keep SKILL.md concise. Put detail in `references/`. State hard rules plainly, for example: never tune on the test split; never cancel AI Runtime runs you did not start; pin `trl==1.15.0` with `unsloth==2026.11.1`; no `mlflow.set_experiment` inside AIR runs.

## Scripts (generic, driven by `task_card.yaml`)
- `scripts/task_adapter.py`: loads the task card; defines the prompt template, the output JSON schema and the metric for a task. Ships with a PHEE adapter as the example.
- `scripts/rewards.py`: reward functions in TRL format (`def f(completions, **kwargs) -> list[float]`). Implement the gated extraction reward from `research/03-dataset-phee.md`: zero for invalid JSON, off-schema output, or any string not found verbatim in the input; otherwise F1 over a one-to-one event matching; both empty = 1.0; predicted empty when gold is not empty = 0.0; no standalone format bonus.
- `scripts/prepare_data.py`: download PHEE V2 from GitHub, convert to chat-format JSONL with train/dev/test, write a data card with counts and a near-duplicate check between splits.
- `scripts/train_sft.py`, `scripts/train_grpo.py`: TRL + Unsloth, LoRA, configured by CLI args or env vars, MLflow logging, save adapters to a UC volume path.
- `scripts/evaluate.py`: batch inference with vLLM, metrics, MLflow logging; also evaluates an FMAPI model for baselines.
- `scripts/probe.py`: the 30-step probe that reports memory and throughput.
- `scripts/workloads/*.yaml`: AI Runtime workload templates for probe, SFT, GRPO and eval, using `databricks_ai_v6`, `max_retries: 0`, H100 by default with A10 notes.

## Constraints for this session
- Do not submit any AI Runtime runs and do not create Databricks resources in this session. Only author files and run local CPU tests.
- Write tests in `skill/custom-llm-builder/tests/` for `rewards.py` and `task_adapter.py`: invalid JSON, a string not in the input, both empty, empty vs gold, copying the whole sentence into every field, duplicate events, partial matches. Run them with `uv run --with pytest pytest skill/custom-llm-builder/tests -q` and make them pass.
- Do not install anything globally. Do not print tokens.
- Write prose in plain, simple sentences.

When done, reply with: the file tree, the test result line, and any open questions or doubts about the research inputs.
