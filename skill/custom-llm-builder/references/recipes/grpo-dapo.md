# GRPO and DAPO

Use RL only after rollouts have a useful valid-format rate and nonzero reward variance. Start from SFT when the base model is cold. TRL 1.15 defaults to a DAPO-style loss; use `beta=0`, lower clip near 0.2, upper clip near 0.28, and dynamic sampling of mixed-reward groups when supported. Dr. GRPO is a comparison option.

Start with 4 generations, completion length 256 to 512, device batch size 1, accumulation 4, LoRA rank 16, alpha 32, and gradient checkpointing. These are probe values, not facts. H100 is preferred for colocated vLLM. A10 may require 4-bit weights and fewer generations.

Track reward, `frac_reward_zero_std`, completion clipped ratio, entropy, KL if enabled, memory, and step time. Before trusting RL, generate identical prompts from the base and tuned adapters and confirm scores differ. This detects the Unsloth shared-weight LoRA injection failure.
