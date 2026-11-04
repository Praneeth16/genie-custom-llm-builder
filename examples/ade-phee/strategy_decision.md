# ADE-PHEE Strategy Decision

All quality measurements use the frozen 961-row development split. The test split was not scored.

## Final Candidate Freeze

Frozen at `2026-11-04T12:14:19+05:30`, before any test scoring. GRPO Arm B (0.7818 dev strict event F1) and SFT-full v2 (0.7817) are tied because their 0.0001 gap is far below the 0.0080 noise reference. Per the pre-declared rule that a dev tie goes to the cheaper model, the final candidate is **SFT-full v2** at `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/adapters/full_v2/`. GRPO Arm B is a pre-declared secondary test result only and is not used for selection. This decision is immutable with respect to the frozen test split.

## Signals

| Signal | Value | Rule triggered |
|---|---:|---|
| Gold labels present | true | Supervised learning is available. |
| Closed schema | true | Favors LoRA SFT for verifiable extraction. |
| Train size | 2,898 | Above the approximate “fewer than 1,000” starting point; measure learning curves rather than treating the threshold as universal. |
| Base valid-format rate | 0.759766 | Does not trigger the below-0.5 mandatory SFT-warm-start rule. |
| Mean rollout reward | 0.218148 | Below the corrected task-card target of 0.7655, so prompting does not meet the stop rule. |
| Mean within-prompt reward standard deviation | 0.033018 | Some variation exists, but it is sparse. |
| Prompts with zero reward standard deviation | 0.750000 | Exceeds 0.6, so GRPO has little measured learning signal and is not the first method. |
| Empty-output rate | 0.275391 | Material base failure mode for SFT to address. |
| Teacher token logprobs available | false | On-policy GKD is not selected. |

## Decision

Start with LoRA SFT on the gold training labels. After parser fixes, the frontier five-shot baseline reaches 0.7155 gated strict event F1, below the 0.7655 target, while the base model reaches only 0.3007 five-shot. Gold labels and a closed schema make SFT the most direct first intervention, and the 0.75 zero-variance fraction argues against spending the first training budget on GRPO.

## Planned Comparison

Measure 500-example, 2,000-example, and full-training-data LoRA SFT curves when budget permits. If SFT produces valid, varied outputs, compare full-data SFT with SFT followed by GRPO on the same frozen development protocol at matched total accelerator time and matched evaluation decoding.

Change the plan if either of these occurs:

- Stop at SFT if it reaches 0.7655 gated strict event F1 and passes throughput and cost targets.
- Test SFT+GRPO only if resampled SFT rollouts reduce the zero-reward-standard-deviation fraction to at most 0.6 and tuned rollouts score differently from base rollouts on identical prompts. Otherwise prefer SFT or off-policy teacher distillation rather than RL.

## After SFT

Run `352574233679407` completed on one H100. All curve points used seed 42, rank 32, alpha 32, learning rate `2e-4`, completion-only loss, and three epochs; holding epochs constant produced 189, 750, and 1,089 optimizer steps. The test split was not scored.

| Train rows | Dev strict F1 | Valid rate | PHEE EM | PHEE token F1 | Empty rate | Throughput (rows/s) | Train accelerator minutes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.7673 | 0.9906 | 0.6534 | 0.7748 | 0.000000 | 83.53 | 4.25 |
| 2,000 | 0.7734 | 0.9938 | 0.6611 | 0.7807 | 0.000000 | 82.13 | 13.70 |
| 2,898 | 0.7814 | 0.9896 | 0.6694 | 0.7886 | 0.000000 | 77.93 | 19.30 |

Evaluation consumed another 2.77, 1.79, and 1.44 accelerator minutes respectively. The 30-step probe passed at 2.76 seconds per step, 11.52% peak H100 memory, finite loss, and 2.64 accelerator minutes. Package setup consumed 0.49 accelerator minutes; torch remained `2.11.0+cu130`. The image's unrelated `tabfm`/`typeguard` conflict was recorded but did not mask training-stack conflicts.

### Corrected SFT Evidence

The original SFT evaluations and rollout signals were invalid because the metric parser did not strip the empty Qwen3 reasoning prefix (`<think>\n\n</think>\n\n`). Re-scoring saved predictions selects the full-data adapter at 0.7814 dev strict event F1, above the corrected 0.7655 target and the corrected Claude 5-shot baseline of 0.7155. Claude FMAPI predictions were also re-scored after extracting text from `reasoning` plus `text` content blocks. Base Qwen zero-shot predictions were overwritten in the shared AIR run, so its originally reported score remains unchanged and could not be re-scored. The SFT signals from run `352574233679407` used the broken parser and must not be used for the RL gate.

### Next Decision

Recompute signals from the SFT-full adapter with the fixed parser. Run matched-compute continued SFT versus GRPO only if the new zero-standard-deviation fraction is at most 0.6 and tuned rollout rewards differ from base rewards on identical prompts.

## RL Experiment

AI Runtime run `823199832587319` completed successfully on one H100. The fixed-parser signals used the same first 64 development rows, eight samples per prompt, temperature 1.0, and seed 42. The test split remained untouched.

| Model | Dev strict F1 | Valid rate | PHEE EM | Token F1 | Precision | Recall | Accelerator minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT full | 0.7809 | 0.9896 | 0.6687 | 0.7880 | 0.7859 | 0.7810 | 19.30 train + 1.75 eval |
| Arm A: continued SFT | — | — | — | — | — | — | Not run (gate failed) |
| Arm B: GRPO | — | — | — | — | — | — | Not run (gate failed) |

### Signals and Gate

The SFT-full rollouts had valid-format rate 0.0039, mean reward 0.0031, mean within-prompt reward standard deviation 0.0082, zero-standard-deviation fraction 0.96875, and empty-output rate 0. The base rollouts had valid-format rate 0.7598, mean reward 0.2181, mean within-prompt reward standard deviation 0.0330, zero-standard-deviation fraction 0.75, and empty-output rate 0.2754. Tuned reward vectors differed from base reward vectors, but the SFT-full zero-standard-deviation fraction exceeded the 0.6 ceiling, so the gate failed.

The workload therefore skipped dynamic prompt sampling, the GRPO probe, both matched-compute arms, and their evaluations as specified. No Unsloth or plain-TRL GRPO path was used. The strongest alternative is to retain SFT full; if further improvement is required, use continued supervised fine-tuning or off-policy Claude distillation rather than GRPO under this sampling setup.

### Conclusion

GRPO did not demonstrate an improvement over continued SFT at matched compute because the prerequisite learning-signal gate failed and neither arm was run. The GRPO-minus-continued-SFT delta is therefore unavailable, and there is no evidence of a gain larger than the 0.0080 dev strict-F1 difference between SFT 2,000 (0.7734) and SFT full (0.7814). By dev score only, the final candidate for the untouched test split is the SFT-full adapter at `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/adapters/full/`.

### Sampling Diagnostic Follow-up

AI Runtime run `81591482554133` tested whether the failed gate was caused by unconstrained sampling. Each setting generated eight samples for the same 64 development prompts and saved all 512 texts. An invalid sample was assigned to exactly one cause in this order: non-empty reasoning block, trailing text after a parseable JSON value, broken JSON, non-verbatim span, or other schema failure.

| Setting | Temperature | Top-p | Top-k | Valid rate | Mean reward | Mean reward std | Zero-std fraction | Trailing text | Reasoning content | Broken JSON | Non-verbatim | Other | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| (a) Full vocabulary | 1.0 | 1.0 | -1 | 0.0039 | 0.0031 | 0.0082 | 0.96875 | 0 | 0 | 510 | 0 | 0 | Fail |
| (b) Qwen recommended | 0.7 | 0.8 | 20 | 0.4121 | 0.3186 | 0.3703 | 0.015625 | 0 | 0 | 300 | 1 | 0 | Fail |
| (c) Nucleus only | 1.0 | 0.95 | -1 | 0.0117 | 0.0088 | 0.0233 | 0.90625 | 0 | 0 | 506 | 0 | 0 | Fail |

Setting (b) restored strong within-prompt reward variance but missed the required 0.5 valid-rate floor at 0.4121. Settings (c) and (a) failed both thresholds. The captured samples therefore do not support the trailing-text hypothesis: failures were overwhelmingly unparseable responses, often arbitrary leading content around an otherwise plausible answer, rather than text appended after a parseable JSON document.

No setting passed both zero-std fraction at most 0.6 and valid rate at least 0.5, so the workload correctly stopped before dynamic sampling, the GRPO probe, and both 40-minute arms. No GRPO implementation path was used and no matched-compute RL table exists. The final candidate remains SFT full by dev score only; the test split remains untouched.

### Reference-template Retrain

AI Runtime run `520951822433666` retrained SFT full after forcing the upstream Qwen chat template. The saved rendered training example contains the expected system, user, and assistant turns and no `<think>` block. Training used the same 2,898 rows, three epochs, 1,089 optimizer steps, rank 32, alpha 32, learning rate `2e-4`, and seed 42 as the original full-data run. It consumed 19.26 accelerator minutes, reached loss 0.0593, and peaked at 11.64% H100 memory.

| Model | Dev strict F1 | Valid rate | PHEE EM | Token F1 | Precision | Recall | Reasoning-prefix rate | Accelerator minutes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT full v2 | 0.7817 | 0.9896 | 0.6703 | 0.7882 | 0.7859 | 0.7828 | 0.0000 | 19.26 train + 2.75 eval |
| Arm A: continued SFT | 0.7662 | 0.9854 | 0.6539 | 0.7724 | 0.7729 | 0.7653 | 0.0000 | 40.16 train + 2.53 eval |
| Arm B: GRPO | 0.7818 | 0.9886 | 0.6712 | 0.7881 | 0.7852 | 0.7838 | 0.0000 | 40.29 train + 1.71 eval |

| Setting | Temperature | Top-p | Top-k | Valid rate | Mean reward | Mean reward std | Zero-std fraction | Corrupt preamble | Broken JSON | Non-verbatim | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| (b) Qwen recommended | 0.7 | 0.8 | 20 | 0.9980 | 0.7857 | 0.0558 | 0.515625 | 0 | 0 | 1 | Pass |
| (a) Full vocabulary | 1.0 | 1.0 | -1 | 0.9805 | 0.7511 | 0.1205 | 0.203125 | 0 | 1 | 9 | Pass |

Setting (b) passed first and was selected. Dynamic sampling retained 891 of 2,898 training prompts (30.75%). The template fix eliminated the corrupted-special-token preamble and restored both validity and within-prompt reward variance.

Run `528364182692010` reused the 891-prompt filter and used a generation batch of 32 (`per_device_train_batch_size=8`, four accumulation steps, eight generations). The Unsloth 10-step probe passed without fallback: reward 0.8838, `frac_reward_zero_std` 0.15, clipped ratio 0, 13.16 seconds mean step time (7.40 seconds for the final steady-state step), and 87.80% peak H100 memory.

At matched compute, GRPO beat continued SFT by 0.0156 dev strict F1. This is larger than the 0.0080 gap between SFT 2,000 and the original SFT full checkpoint. However, continued SFT regressed by 0.0155 from the SFT-full-v2 starting point, while GRPO improved over that starting point by only 0.00008, far below the noise reference. The honest conclusion is that GRPO protected quality relative to an over-trained SFT control, not that it produced a meaningful improvement over SFT full v2.

The base-versus-tuned rollout check passed: sample texts and reward groups differed, with mean reward 0.7881 for Arm B versus 0.2218 for base, confirming that generation used the GRPO LoRA weights. By dev score only, the final candidate for the untouched test split is Arm B at `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/rl/adapters/grpo/` with 0.7818 strict event F1.

## Test Results

The final candidate freeze above preceded every test run. The frozen 968-row test split was scored once per declared model with the dev prompt builder and greedy decoding. Claude used concurrency 8; the local vLLM runs used one H100. Local vLLM did not emit token-usage telemetry.

| Model | Role | Strict event F1 | Valid rate | PHEE EM | PHEE token F1 | Precision | Recall | Throughput (rows/s) | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5.5 zero-shot | baseline | 0.6351 | 0.9473 | 0.4796 | 0.6394 | 0.6280 | 0.6561 | 2.71 | 475,188 |
| Claude Sonnet 5.5 five-shot | baseline | 0.7083 | 0.9855 | 0.5636 | 0.7173 | 0.7074 | 0.7163 | 2.82 | 1,111,398 |
| Qwen3-4B five-shot | baseline | 0.3037 | 0.5826 | 0.2273 | 0.3102 | 0.3062 | 0.3029 | 185.78 | n/a (local vLLM) |
| SFT-full v2 | final | **0.7856** | **0.9969** | **0.6695** | **0.7927** | **0.7906** | **0.7857** | 79.16 | n/a (local vLLM) |
| GRPO Arm B | secondary | 0.7838 | 0.9969 | 0.6676 | 0.7909 | 0.7878 | 0.7851 | 81.98 | n/a (local vLLM) |

SFT-full v2 remains the shipped model. Its test strict event F1 exceeds the 0.7655 target, and its one-H100 throughput exceeds the 20 sentences/second target. GRPO is reported only as the pre-declared secondary result and does not affect selection.
