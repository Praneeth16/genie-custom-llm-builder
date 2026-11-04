# ADE-PHEE Strategy Decision

All quality measurements use the frozen 961-row development split. The test split was not scored.

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
