# TRL + Unsloth stack (research as of 2026-11-04)

## Versions
- TRL latest: 1.17.1 (30 Oct 2026) ([releases](https://github.com/huggingface/trl/releases/latest)).
- Unsloth: `unsloth==2026.11.1` (3 Nov 2026) ([PyPI](https://pypi.org/project/unsloth/2026.11.1/)) with matching `unsloth_zoo`.
- Unsloth requires `trl>=0.18.2,!=0.19.0,<=1.15.0` and `transformers<=5.17.0`, excluding 5.0.0 and 5.1.0 ([pyproject](https://raw.githubusercontent.com/unslothai/unsloth/main/pyproject.toml)). So pin `trl==1.15.0` when using Unsloth. Do not run `pip install -U trl`.
- vLLM: Unsloth Zoo rejects 0.10.x and 0.14.x with Standby mode; it suggests >=0.11.2 or >=0.15.1 ([vllm_utils.py](https://raw.githubusercontent.com/unslothai/unsloth-zoo/main/unsloth_zoo/vllm_utils.py)). Which vLLM ships in `databricks_ai_v6` is not verified; check at runtime.

## Model
- Primary: `Qwen/Qwen3-4B-Instruct-2507`, Apache-2.0, text only ([HF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)).
- Fallback only: `Qwen/Qwen3.5-4B`. It is a vision-language model with Gated DeltaNet layers ([HF](https://huggingface.co/Qwen/Qwen3.5-4B)). Unsloth says to use transformers v5, 16-bit LoRA, `fast_inference=False` ([guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)). Open crash report: [unsloth #4612](https://github.com/unslothai/unsloth/issues/4612).
- Other: `google/gemma-4-E4B` (Apache-2.0). Unsloth's GRPO notebook targets E2B, not E4B.

## GPUs on Databricks AI Runtime
1xA10 (24 GB), 1xH100 (80 GB), 8xH100. Only 8xH100 is multi-GPU ([AIR overview](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/overview)). H100 is preferred for GRPO with vLLM colocated. A10 works for 4B LoRA SFT and for GRPO with 4-bit weights and few generations.

## TRL GRPOTrainer ([docs](https://huggingface.co/docs/trl/main/en/grpo_trainer))
- `loss_type` default is `dapo`. Also `dr_grpo`, `sapo`, `cispo`.
- vLLM: `use_vllm=True`, `vllm_mode="colocate"` (default, `vllm_gpu_memory_utilization` 0.3) or `"server"`.
- Reward function signature: `def f(completions, **kwargs) -> list[float]`. Extra dataset columns arrive as kwargs. `None` is allowed for rows a reward does not apply to.
- LoRA via `peft_config`. `beta=0.0` by default, so no reference model is loaded.
- `environment_factory` (experimental, transformers>=5.2.0) for multi-turn tool environments.

## Unsloth GRPO
- Standby: set `os.environ["UNSLOTH_VLLM_STANDBY"]="1"` before importing unsloth ([memory-efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)).
- Known TRL 1.x problem: with shared weights, Standby may not wake and LoRA may not be injected into vLLM generation, so rollouts can silently use the base model. Fixes are on Unsloth `main` ([rl_replacements.py](https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/rl_replacements.py)). Always check that tuned rollouts score differently from base rollouts.
- Official notebooks: `Qwen3_(4B)-GRPO.ipynb` (LoRA r 16, 2 generations, 1024/1024 tokens, T4). Not extraction specific. Community JSON-extraction GRPO example: [blog](https://leoniemonigatti.com/blog/fine-tuning-lfm2-5-1-2b-instruct-with-grpo.html).

## Starting hyperparameters (unverified; measure with a 30-step probe)
`num_generations=4` (8 on H100), `max_completion_length` 256 to 512 (watch `completions/clipped_ratio`), `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, LoRA `r=16`, `alpha=32`, gradient checkpointing on. 4-bit only on A10. Step time and LR are not benchmarked anywhere; measure.

## Existing skills to borrow ideas from (not to install)
- Hugging Face `huggingface-llm-trainer` ([SKILL.md](https://github.com/huggingface/skills/blob/main/skills/huggingface-llm-trainer/SKILL.md)). Tied to HF Jobs.
- TRL `trl-training` ([SKILL.md](https://github.com/huggingface/trl/blob/main/skills/trl-training/SKILL.md)).
- Community Unsloth skill ([Orchestra-Research](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/main/03-fine-tuning/unsloth/SKILL.md)).
None of them knows Databricks AI Runtime or chooses between methods. That is the gap this skill fills.
