# Strategy Selection

Measure signals before choosing. Log `strategy`, `strategy_reason`, `train_size`, `valid_format_rate`, `reward_variance`, `gold_labels_present`, and `teacher_logprobs_available` as MLflow tags or metrics.

| Measured signal | First method to test |
|---|---|
| Prompt baseline meets all task-card targets | Stop. Ship the prompt. |
| Fewer than about 100 examples and no training budget | GEPA prompt optimization |
| Gold labels, closed schema, fewer than about 1,000 examples | LoRA SFT |
| Teacher outputs but no gold labels | Off-policy distillation through SFT |
| Open teacher supplies token logprobs and student errors matter | On-policy GKD with `lmbda=1`, `beta=1` |
| Most base rollouts are invalid | SFT warm start before any RL |
| SFT outputs are valid and varied, and reward is verifiable | Compare SFT+GRPO with SFT at matched compute |
| Preference pairs only | Preference optimization may fit style, but is a weak fit for extraction F1 |

No source establishes a universal train-size threshold. For a material decision, measure 500, 2,000, and full-data curves when budget permits.

RL on top of SFT is not proven for extraction. The clinical evidence lacks an SFT-only arm. A gain is established only by a matched-compute comparison on the same frozen development protocol.

Read the selected recipe only:

- [LoRA SFT](recipes/sft-lora.md)
- [Off-policy distillation](recipes/distill-offpolicy.md)
- [On-policy GKD](recipes/gkd-onpolicy.md)
- [GRPO and DAPO](recipes/grpo-dapo.md)
- [GEPA baseline](recipes/gepa-baseline.md)
