Use $custom-llm-builder for the RL experiment in `examples/ade-phee/`. Profile `fe-vm-lakebase-praneeth`. Dev split only; the test split is still untouched.

## What changed since session D (read first)
The project lead found two scoring bugs and fixed them in `skill/custom-llm-builder/scripts/metrics.py` (tests pass):
1. The SFT adapters emit an empty Qwen3 reasoning block (`<think>\n\n</think>\n\n`) before otherwise valid JSON, so every SFT output scored 0. The parser now strips one leading `<think>...</think>` block for every model.
2. Some FMAPI Claude replies arrive as content blocks (`reasoning` + `text`). The parser now reads the text blocks, and `evaluate.py` stores text.
`examples/ade-phee/rescore.py` re-scored the saved dev predictions; results are in `examples/ade-phee/rescored_dev.json`. SFT full = 0.7814 dev strict event F1, Claude 5-shot = 0.7155, so SFT meets the 0.7655 target. The SFT signals from session D were scored with the broken parser and are invalid.

First, update `examples/ade-phee/strategy_decision.md` and the baseline numbers in the task card with the re-scored values, and add one line to `references/air-glue.md` or `references/eval-protocol.md` about both parser issues. Base Qwen zero-shot predictions were overwritten in the shared AIR run, so that number stays as originally reported; say so.

## Goal
Use `create_goal` for this session with the objective "Answer whether GRPO improves PHEE extraction over SFT at matched compute, with dev evidence" if the tool is available to you; report whether it was.

## One AI Runtime workload on GPU_1xH100
All steps start from the SFT-full adapter at `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/adapters/full/`. Each step writes JSON results to `.../runs/rl/` and logs MLflow tags `phase=rl`, `step=<name>`. `max_retries: 0`, timeout 300 minutes, idempotency key, dry-run first. Keep the filtered `pip check` and torch guard from `run_sft_phase.py`. The image has vLLM 0.24.0; Unsloth's GRPO path has only been checked against older vLLM lines, so a short GRPO probe comes before anything long.
1. Signals on the SFT-full adapter with the fixed parser: the same 64-row dev subset, 8 samples, temperature 1.0. Also the base-vs-tuned rollout check.
2. Gate: continue only if the zero-std fraction is at most 0.6 and tuned rollouts differ from base. If the gate fails, write the numbers, skip to step 6, and propose the best alternative.
3. GRPO prompt set: sample 8 completions per train prompt from the SFT-full adapter with vLLM and keep the prompts with non-zero reward std (dynamic sampling). Record how many prompts survive.
4. GRPO probe: 10 steps (DAPO loss, `beta=0`, clip 0.2/0.28, 8 generations, max completion 256 tokens, LR 5e-6, LoRA continued from SFT-full). Check reward is non-zero, `frac_reward_zero_std`, clipped ratio, step time and memory. If the probe fails on the Unsloth plus vLLM 0.24 combination, retry once with plain TRL GRPO (PEFT, `use_vllm=True`, colocate) without Unsloth, and record which path worked.
5. Matched compute, 40 accelerator minutes per arm, both from SFT-full:
   - Arm A, continued SFT: one more pass on full train at LR 5e-5, stopped at 40 minutes.
   - Arm B, GRPO: on the filtered prompt set, stopped at 40 minutes.
   Save both adapters under `.../runs/rl/adapters/{sft_plus,grpo}/`.
6. Evaluate SFT-full, Arm A and Arm B on all 961 dev rows with `evaluate.py` (temperature 0, same prompt). Use distinct MLflow run names per model so predictions are not overwritten.

## After the run
Add a "RL experiment" section to `strategy_decision.md` with a table (model, dev strict F1, valid rate, PHEE EM, token F1, precision, recall, accelerator minutes) and a plain conclusion: did GRPO beat continued SFT at matched compute, by how much, and whether the difference is larger than the noise between SFT 2,000 and SFT full. Pick the final candidate for the test split by dev score only, and write the choice in the decision file.

Reply with the table, the signals, the gate result, the GRPO path used, the final candidate, the AIR run id, MLflow links and any bugs hit.
