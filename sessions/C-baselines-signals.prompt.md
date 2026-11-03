Use $custom-llm-builder for phase 2 (baselines) and phase 3 (strategy signals) of `examples/ade-phee/`. Read `examples/ade-phee/task_card.yaml` first. Use the `databricks-ai-runtime` skill for workload mechanics. Profile: `fe-vm-lakebase-praneeth`.

All numbers in this session are on the dev split. Do not score the test split.

## A. Frontier baselines (run locally, not on AI Runtime)
- Endpoint `databricks-claude-sonnet-5-5`. Zero-shot and 5-shot (fixed seed 42, examples from train) on all 961 dev rows, using the shared prompt builder.
- Claude 5.x endpoints reject `temperature`; send none.
- Add a `--concurrency N` option to `evaluate.py` for the FMAPI path (thread pool, default 8, retry with backoff on 429 and 5xx) so this finishes in reasonable time. Add a test with a fake client.
- Log to MLflow in the workspace: set `MLFLOW_TRACKING_URI=databricks`, `DATABRICKS_CONFIG_PROFILE=fe-vm-lakebase-praneeth`, and use experiment `/Workspace/Users/praneeth.paikray@databricks.com/custom_llm/ade_phee` (setting the experiment is fine locally; only AIR runs must not call `set_experiment`). Tag runs with `phase=baseline`, `split=dev`, `model`, `shots`.
- Record token usage if the response returns it, and compute cost per 1,000 sentences from list prices if you can find them in the docs; otherwise record tokens only and say so.

## B. One AI Runtime run on GPU_1xH100 for the base small model
Put these steps in one workload so we pay the provisioning wait once:
1. Print versions of torch, transformers, trl, vllm, unsloth, peft. Do not install unsloth or a different trl in this run (not needed for inference); use the image's vLLM.
2. Base `Qwen/Qwen3-4B-Instruct-2507` zero-shot and 5-shot on dev with `evaluate.py` (vLLM, temperature 0).
3. `signals.py` on a fixed 64-row dev subset with 8 samples per prompt at temperature 1.0.
Data comes from `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/`. Write outputs under `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/`. Experiment directory `/Workspace/Users/praneeth.paikray@databricks.com/custom_llm`, experiment name `ade_phee`. `max_retries: 0`, timeout 120 minutes. Run `--dry-run` first, then submit. Use an idempotency key.
While it provisions (about 30 minutes) do part A. Check the run with `databricks air get` and short log snapshots (`timeout 30 databricks air logs <id> --tail 80`); never stream logs without a timeout. Never cancel runs you did not start.

## C. Strategy decision
When A and B are done, fill in the numeric quality target in the task card, then apply `references/strategy-selection.md` to the measured signals and write `examples/ade-phee/strategy_decision.md`: the signal values, the rule each one triggered, the chosen first method, the planned comparison (for example SFT vs SFT then GRPO at matched compute), and what result would change the plan. Log the decision as MLflow tags on a `phase=strategy` run.

Reply with a baseline table (model, shots, gated strict event F1, valid-format rate, PHEE EM, PHEE token F1, empty-output rate, throughput, tokens or cost), the signals, the decision, the AIR run id, MLflow run links, and any bugs you hit in Genie, the CLI or AI Runtime.
