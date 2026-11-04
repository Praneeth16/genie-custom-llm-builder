# AI Runtime Glue

Delegate workload field details and lifecycle commands to `databricks-ai-runtime`. Confirm fields with CLI help and dry-run before submission. `code_source.root_path` is relative to the YAML file.

Use `environment.version: databricks_ai_v6` (the AI image with torch, transformers, TRL and vLLM). Do not use `"6"`: it selects a root Python without torch or vLLM. Use `max_retries: 0` and H100 by default. Set `VLLM_USE_FLASHINFER_SAMPLER=0`; set `UNSLOTH_VLLM_STANDBY=1` before importing Unsloth when using Standby. Check installed vLLM at runtime.

Installing Unsloth can replace the torch build preinstalled by `databricks_ai_v6`. The probe workload prints torch, transformers, TRL, vLLM, and Unsloth versions before importing the training script. Stop if torch or CUDA compatibility changes unexpectedly.

Workload templates contain explicit placeholder `env_variables`. Replace them at dry-run or submission time with repeated overrides such as `--override env_variables.TRAIN_DATA=/Volumes/<catalog>/<schema>/<volume>/train.jsonl`.

AIR provides one shared `MLFLOW_RUN_ID`. Use plain `mlflow.start_run()` and set the experiment in workload YAML. Never call `mlflow.set_experiment` in the run. MLflow params are immutable, so distinguish multi-step work with tags (`phase`, `step`) or step-suffixed metric/artifact names instead of changing params. Names use only letters, digits, hyphens, and underscores.

If the local sandbox blocks profile keyring access, mint a short-lived token with `databricks auth token --profile <profile>` inside the same shell command and pass it only through `DATABRICKS_TOKEN`; never print or persist it.

Save adapters under a configured `/Volumes/<catalog>/<schema>/<volume>` path. Register the selected model with the UC registry after evaluation. Convert a validated evaluation workload to a DAB for scheduled batch scoring. A live serving endpoint is optional.

After installing pins, check that torch is unchanged first, then run `pip check` but fail only on lines whose first package is in the training stack (torch, transformers, trl, vllm, unsloth, unsloth_zoo, peft, accelerate, datasets, bitsandbytes, xformers). The `databricks_ai_v6` image already has unrelated conflicts (for example `tabfm` wants `typeguard<3`), so a whole-environment `pip check` always fails.

Unsloth may load a mirrored repo (for example `unsloth/Qwen3-4B-Instruct-2507`) whose chat template differs from the upstream model. That mirror adds an empty `<think></think>` block to every assistant turn, so SFT learns a special-token preamble: greedy decoding hides it, but sampling (and therefore GRPO) breaks it by swapping in `<tool_call>` and similar tokens. `train_sft.py` and `train_grpo.py` call `chat_template.use_reference_chat_template` to force the upstream template and fail if the rendering differs. Always check sampled outputs, not only greedy ones, before RL.
