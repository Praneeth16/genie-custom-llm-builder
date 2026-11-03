# ADE-PHEE Strategy Decision

All quality measurements use the frozen 961-row development split. The test split was not scored.

## Signals

| Signal | Value | Rule triggered |
|---|---:|---|
| Gold labels present | true | Supervised learning is available. |
| Closed schema | true | Favors LoRA SFT for verifiable extraction. |
| Train size | 2,898 | Above the approximate “fewer than 1,000” starting point; measure learning curves rather than treating the threshold as universal. |
| Base valid-format rate | 0.759766 | Does not trigger the below-0.5 mandatory SFT-warm-start rule. |
| Mean rollout reward | 0.218148 | Below the task-card target of 0.761164, so prompting does not meet the stop rule. |
| Mean within-prompt reward standard deviation | 0.033018 | Some variation exists, but it is sparse. |
| Prompts with zero reward standard deviation | 0.750000 | Exceeds 0.6, so GRPO has little measured learning signal and is not the first method. |
| Empty-output rate | 0.275391 | Material base failure mode for SFT to address. |
| Teacher token logprobs available | false | On-policy GKD is not selected. |

## Decision

Start with LoRA SFT on the gold training labels. The frontier five-shot baseline reaches 0.711164 gated strict event F1, below the 0.761164 target, while the base model reaches only 0.300729 five-shot. Gold labels and a closed schema make SFT the most direct first intervention, and the 0.75 zero-variance fraction argues against spending the first training budget on GRPO.

## Planned Comparison

Measure 500-example, 2,000-example, and full-training-data LoRA SFT curves when budget permits. If SFT produces valid, varied outputs, compare full-data SFT with SFT followed by GRPO on the same frozen development protocol at matched total accelerator time and matched evaluation decoding.

Change the plan if either of these occurs:

- Stop at SFT if it reaches 0.761164 gated strict event F1 and passes throughput and cost targets.
- Test SFT+GRPO only if resampled SFT rollouts reduce the zero-reward-standard-deviation fraction to at most 0.6 and tuned rollouts score differently from base rollouts on identical prompts. Otherwise prefer SFT or off-policy teacher distillation rather than RL.

## After SFT

Run `352574233679407` completed on one H100. All curve points used seed 42, rank 32, alpha 32, learning rate `2e-4`, completion-only loss, and three epochs; holding epochs constant produced 189, 750, and 1,089 optimizer steps. The test split was not scored.

| Train rows | Dev strict F1 | Valid rate | PHEE EM | PHEE token F1 | Empty rate | Throughput (rows/s) | Train accelerator minutes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 83.53 | 4.25 |
| 2,000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 82.13 | 13.70 |
| 2,898 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 77.93 | 19.30 |

Evaluation consumed another 2.77, 1.79, and 1.44 accelerator minutes respectively. The 30-step probe passed at 2.76 seconds per step, 11.52% peak H100 memory, finite loss, and 2.64 accelerator minutes. Package setup consumed 0.49 accelerator minutes; torch remained `2.11.0+cu130`. The image's unrelated `tabfm`/`typeguard` conflict was recorded but did not mask training-stack conflicts.

### SFT Signals

The curve has no meaningful best checkpoint because all three adapters tie at zero and every decoded output is invalid. The workload's deterministic tie-break selected the 500-row adapter for signals. On the fixed 64-row subset with eight samples at temperature 1.0, it had valid-format rate 0, mean reward 0, mean within-prompt reward standard deviation 0, zero-standard-deviation fraction 1.0, and empty-output rate 0. Its reward vectors differ from the base vectors, but only because tuning collapsed every rollout to an invalid, zero-reward output. The base comparison retained mean reward 0.218148 and zero-standard-deviation fraction 0.75.

### Next Decision

Do not run SFT+GRPO: the SFT rollout zero-standard-deviation fraction is 1.0, above the 0.6 gate. First diagnose the training/evaluation contract—especially chat-template/completion serialization and LoRA decoding—using saved predictions and a few adapter generations. If SFT validity cannot be restored, prefer off-policy distillation from Claude on train inputs. Higher temperature or more generations will not repair the observed all-invalid collapse.
