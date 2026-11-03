Use $custom-llm-builder for phase 4 (train) of `examples/ade-phee/`. Read `examples/ade-phee/task_card.yaml` and `examples/ade-phee/strategy_decision.md` first. Profile `fe-vm-lakebase-praneeth`. Dev split only; do not score test.

## 1. Fold what you learned into the skill first
Session C hit four AI Runtime and sandbox issues (`databricks_ai_v6` vs `"6"`, `VLLM_USE_FLASHINFER_SAMPLER=0`, MLflow params are immutable inside the shared AIR run so use tags or step-suffixed names, keyring blocked in the sandbox so use a short-lived `databricks auth token` inside the command). Add them to `skill/custom-llm-builder/references/air-glue.md` and fix the workload templates so the next user does not hit them. Keep it short.

## 2. One AI Runtime workload on GPU_1xH100 that runs, in order
Each step must write a small JSON result to `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/` and log to MLflow with tags `phase=train` and `step=<name>`, so a failure part way still leaves usable results.
1. Install pins `trl==1.15.0` and `unsloth==2026.11.1` and print torch, transformers, trl, vllm, unsloth, peft versions. If the pins conflict with the image's torch, stop and write the error; do not try other versions blindly.
2. Probe: `train_sft.py` for 30 steps on full train. Gate inside the script: abort the workload if peak memory is above 90% of the GPU or the loss is NaN. Record seconds per step.
3. LoRA SFT learning curve: 500, 2,000 and full train (fixed seed, same hyperparameters: rank 32, alpha 32, LR 2e-4, completion-only loss; pick epochs so the full run is about 3 epochs, and train the smaller sets for the same number of optimizer steps or the same epochs, and say which). Save each adapter under `.../runs/sft/adapters/<size>/`.
4. Evaluate each adapter on all 961 dev rows with `evaluate.py` (vLLM, temperature 0, zero-shot prompt, same decoding as the baselines). Record throughput.
5. Run `signals.py` on the best SFT adapter (same 64-row dev subset, 8 samples, temperature 1.0), and also check that tuned rollouts score differently from base rollouts on the same prompts.
6. Record total accelerator time per step so we can match compute later.
`max_retries: 0`, timeout 240 minutes, idempotency key, `--dry-run` first. Check progress with `databricks air get` and short log snapshots only (`timeout 30 ... --tail 80`). Never cancel runs you did not start.

## 3. After the run
Update `examples/ade-phee/strategy_decision.md` with an "After SFT" section: the learning curve table (size, dev strict F1, valid rate, PHEE EM, PHEE token F1, empty rate, throughput, accelerator minutes), the SFT signals, and the next decision. Note this change to the plan from the project lead: the SFT vs SFT+GRPO comparison exists to answer whether RL helps extraction, so run it if the RL signal gate passes (zero-std fraction at most 0.6 on SFT rollouts and tuned rollouts differ from base), even if SFT already meets the target. If the gate fails, say so and propose the best alternative (for example off-policy distillation from Claude on train inputs, or sampling at higher temperature or more generations to get reward variance).

Reply with the learning curve table, SFT signals, the next decision, the AIR run id, MLflow links, and any bugs hit.
