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
