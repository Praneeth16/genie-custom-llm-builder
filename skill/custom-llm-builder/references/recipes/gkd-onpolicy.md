# On-Policy GKD

Use only with an open-weight teacher that exposes token logprobs. In TRL, start with `GKDTrainer`, `lmbda=1`, and `beta=1`. The student generates its own rollouts, and the reverse-KL signal uses teacher and student token probabilities.

Compare against off-policy teacher SFT and gold SFT when available. Probe memory and step time before the full run. Do not substitute an FMAPI teacher without logprobs.
