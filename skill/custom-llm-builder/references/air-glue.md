# AI Runtime Glue

Delegate workload field details and lifecycle commands to `databricks-ai-runtime`. Confirm fields with CLI help and dry-run before submission. `code_source.root_path` is relative to the YAML file.

Use `databricks_ai_v6`, `max_retries: 0`, and H100 by default. Check installed vLLM at runtime. Unsloth Standby rejects some vLLM lines; known safe families are at least 0.11.2 or 0.15.1. Set `UNSLOTH_VLLM_STANDBY=1` before importing Unsloth when using Standby.

Installing Unsloth can replace the torch build preinstalled by `databricks_ai_v6`. The probe workload prints torch, transformers, TRL, vLLM, and Unsloth versions before importing the training script. Stop if torch or CUDA compatibility changes unexpectedly.

Workload templates contain explicit placeholder `env_variables`. Replace them at dry-run or submission time with repeated overrides such as `--override env_variables.TRAIN_DATA=/Volumes/<catalog>/<schema>/<volume>/train.jsonl`.

AIR provides `MLFLOW_RUN_ID`. Use plain `mlflow.start_run()` and set the experiment in workload YAML. Never call `mlflow.set_experiment` in the run. Names use only letters, digits, hyphens, and underscores.

Save adapters under a configured `/Volumes/<catalog>/<schema>/<volume>` path. Register the selected model with the UC registry after evaluation. Convert a validated evaluation workload to a DAB for scheduled batch scoring. A live serving endpoint is optional.
