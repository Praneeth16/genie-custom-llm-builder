# Databricks facts and gotchas (verified in this workspace)

## Workspace
- CLI profile: `fe-vm-lakebase-praneeth` (host `https://fevm-serverless-lakebase-praneeth.cloud.databricks.com`). Serverless only; classic clusters are rejected; the GPU pools API is disabled.
- Catalog: `serverless_lakebase_praneeth_catalog`. Use a new schema `custom_llm` with a volume `artifacts`.
- This is a shared demo workspace. Do not read, list or touch other schemas, apps or AIR runs. One unrelated A10 AIR run may be running; never cancel runs you did not start.
- Frontier models on FMAPI: `databricks-claude-sonnet-5-5`, `databricks-claude-opus-5-5`. Claude 5.x endpoints reject the `temperature` parameter; send none. They return no logprobs, so they can only be off-policy teachers.

## AI Runtime (`databricks air`)
- Use the built-in `databricks-ai-runtime` skill for YAML fields; `databricks air run -h config.<field>` is the source of truth.
- `GPU_1xH100` and `GPU_1xA10` have both run in this workspace before. `environment.version: databricks_ai_v6` has torch, transformers, TRL and vLLM preinstalled.
- `code_source.root_path` is resolved relative to the YAML file's folder.
- Provisioning waits about 30 minutes before the script starts. Plan few, well-chosen runs.
- AIR exports `MLFLOW_RUN_ID`. Calling `mlflow.set_experiment(...)` then `start_run()` fails with "active experiment ID does not match environment run ID". Use plain `mlflow.start_run()` and set the experiment in the YAML (`experiment_name`, `mlflow_experiment_directory`).
- Run and experiment names allow only letters, digits, `-` and `_`.
- `databricks air logs <id>` streams forever; snapshot with `timeout 30 databricks air logs <id> --tail 80`.
- Set `max_retries: 0` while developing (the default is 3, which repeats failures).
- Validate with `databricks air run --file <yaml> --dry-run` before submitting.
- `databricks air convert-to-dabs` turns a workload YAML into a bundle (use for the scheduled batch-scoring job).

## Reference workload that worked (H100, TRL SFT, UC register)
From `02-assets/databricks-agent-loop/train/workload.yaml`:
```yaml
experiment_name: agent-loop-fine-tune
mlflow_experiment_directory: /Workspace/Shared/databricks-agent-loop
mlflow_run_name: qwen3-5-0-8b-exchange-sft
compute:
  accelerator_type: GPU_1xH100
  num_accelerators: 1
environment:
  version: databricks_ai_v6
code_source:
  root_path: .
command: python $CODE_SOURCE_PATH/train.py
timeout_minutes: 90
max_retries: 0
env_variables:
  LOOP_VOLUME: /Volumes/<catalog>/<schema>/artifacts
```
The matching `train.py` used `trl.SFTTrainer`, then `mlflow.register_model(info.model_uri, UC_MODEL, env_pack="databricks_model_serving")` with `mlflow.set_registry_uri("databricks-uc")`.

## Serving
A custom LLM on a GPU_MEDIUM serving endpoint restarted about 4 minutes after loading, twice, with no vLLM logs (open issue). The same entrypoint worked on AIR. Ship batch scoring on AIR; treat a live endpoint as optional.

## Local environment
- Python dependencies through `uv`. `uv.lock` is gitignored because the lock re-resolves through an internal proxy.
- Never print or copy tokens. Use `databricks auth token --profile fe-vm-lakebase-praneeth` only inside commands that need it.
