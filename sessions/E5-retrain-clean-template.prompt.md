Project lead review of the diagnostic samples: the broken JSON is a corrupted preamble, not broken JSON content. Samples look like `<tool_call>\n\n</tool_call>\n\n{"events": [...]}` and `<think>\n\n</tool_call>\n\n{...}`. Root cause, confirmed locally: Unsloth loads `unsloth/Qwen3-4B-Instruct-2507`, whose chat template adds an empty `<think></think>` block to every assistant turn, while the upstream Qwen template (non-thinking model) does not. SFT learned the special-token preamble; greedy decoding hides it, sampling swaps in look-alike special tokens.

I fixed it: `scripts/chat_template.py` forces the upstream template and fails if rendering differs; `train_sft.py` and `train_grpo.py` call it; tests pass; `air-glue.md` and `sessions/BUGS.md` describe it.

You are approved for one more H100 workload that runs, in order:
1. Retrain SFT-full with the fixed template (same data and hyperparameters as session D's full run). Save to `.../runs/sft/adapters/full_v2/`. In the step result, record the rendered text of one training example so we can see there is no `<think>` block.
2. Evaluate SFT-full v2 on all 961 dev rows (temperature 0). Report the reasoning-prefix rate too.
3. Signals on SFT-full v2 at settings (b) T=0.7/top-p 0.8/top-k 20 and (a) T=1.0/top-p 1.0, saving texts and the invalid classification.
4. Gate: setting (b) first, then (a): valid rate at least 0.5 and zero-std fraction at most 0.6.
5. If a setting passes, run the original session E steps 3 to 6 from SFT-full v2 (dynamic-sampling prompt filter, 10-step GRPO probe, 40-minute matched-compute arms for continued SFT and GRPO, dev evals of SFT-full v2, Arm A and Arm B), using that sampling setting for the rollouts and the filter. Keep the Unsloth to plain-TRL fallback rule.
Same rules: dry-run, new idempotency key, `max_retries: 0`, short log snapshots, test untouched, never cancel runs you did not start. If a browser login is requested, stop and say so. Then update the RL experiment section of `strategy_decision.md`, pick the final candidate by dev score, and reply with tables, gate result, GRPO path used, run ids and links.
