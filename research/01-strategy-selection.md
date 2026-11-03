# Post-training strategy selection (research as of 2026-11-04)

Source: web research. 2026 papers marked (snippet) were read from search snippets only. No direct SFT-vs-RL head-to-head on adverse drug event (ADE) extraction was found.

## Decision table: task signal to method

| Signal | Method |
|---|---|
| Gold labels exist, closed JSON schema, under ~1k examples | LoRA SFT first. LoRA rank ~32, LR ~10x the full fine-tune LR, target MLP and attention ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)) |
| Only frontier-teacher outputs, no gold labels | Off-policy SFT on teacher outputs (baseline) |
| Open-weight teacher with logprobs, and the student should fix its own errors | On-policy distillation: student rollouts, reverse KL, token advantage log pi_T - log pi_s ([Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/)). In TRL: `GKDTrainer` with `lmbda=1`, `beta=1` ([TRL GKD](https://huggingface.co/docs/trl/gkd_trainer)) |
| SFT model gives valid, varied outputs and F1 is verifiable | GRPO after SFT. Dr. GRPO (no length or std normalization) ([Dr. GRPO](https://arxiv.org/html/2503.20783v2)) and DAPO-style loss with clip-higher and dynamic sampling of mixed-reward groups ([DAPO](https://arxiv.org/html/2503.14476v1)). GSPO and CISPO are alternatives, not clear winners ([GSPO](https://arxiv.org/html/2507.18071v2), [CISPO](https://arxiv.org/html/2506.13585v1)) |
| Most rollouts fail the format (cold start) | SFT warm start before RL ([SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)) |
| Goal is pass@1 and frontier zero-shot is already high | Expect RL to improve reliability, not coverage ([Yue et al.](https://arxiv.org/abs/2504.13837)) |
| Only preference pairs (style, not F1) | DPO, SimPO, ORPO, KTO. Weak fit for extraction ([comparison](https://arxiv.org/html/2405.14734v3)) |
| No training budget, under ~100 examples | Prompt optimization (GEPA). Beat GRPO on 5 of 6 tasks, +5.9 points on average, but none were extraction tasks ([GEPA](https://arxiv.org/html/2507.19457v2)) |
| Want an SFT variant that generalizes better | Dynamic Fine-Tuning reweights SFT loss by token probability ([DFT](https://arxiv.org/html/2508.05629v1)). One paper only |

## Evidence for extraction tasks
- SFT cold start helps RL in clinical IE: MEDIQA-SYNUR dev F1 0.803 with SFT then GRPO vs 0.620 with GRPO only ([ClinicalNLP 2026](https://aclanthology.org/2026.clinicalnlp-1.18.pdf)). This paper has no SFT-only arm.
- Clinical ADE SFT baselines are strong: F1 0.86 for fine-tuned Mistral-7B on PsyTAR ([Hosseini et al.](https://journals.sagepub.com/doi/10.3233/SHTI250947)); ADE relation F1 0.704 ([JMIR 2025](https://formative.jmir.org/2025/1/e71949/)).
- F1-based rewards with a schema term are used in ReasoningNER ([arXiv 2511.11978](https://arxiv.org/html/2511.11978v1)) and R2GRPO ([arXiv 2505.22068](https://arxiv.org/html/2505.22068v1)).
- (snippet) Nie 2026 ([arXiv 2605.22731](https://arxiv.org/html/2605.22731v1)): Qwen3-0.6B LoRA, mild SFT matched on-policy distillation and beat on-policy RL on GSM8K.
- (snippet) Xu et al., EMNLP 2026 Findings ([link](https://aclanthology.org/2026.findings-emnlp.639/)): RL on the teacher then transfer did worse than direct GRPO on the student before an OPD step.

Conclusion: RL on top of SFT is not proven for extraction. Test it against SFT at matched compute. Do not assume it helps.

## Recommended default recipe for ADE extraction
1. Baselines on a frozen held-out split: frontier zero-shot and few-shot. Score strict event F1, span correctness, negation and temporality errors, cost and latency.
2. LoRA SFT on gold examples (plus teacher-labeled examples if useful). Keep legitimate empty-ADE examples.
3. Optional on-policy distillation if an open-weight teacher with logprobs is available.
4. GRPO from the SFT checkpoint with the gated reward. Compare against SFT-only at matched compute.

## RL settings
- DAPO removes the KL penalty and uses asymmetric clipping (0.2 / 0.28) to avoid entropy collapse.
- Watch zero-std groups (TRL metric `frac_reward_zero_std`). Groups where every sample gets the same reward give no learning signal.
- LoRA Without Regret: rank 1 matched full fine-tuning in RL; LoRA LR about 10x full; keep alpha fixed when comparing ranks.

## Data size
No source gives size thresholds. Sweep 500, 2k and full train and report the curve.
