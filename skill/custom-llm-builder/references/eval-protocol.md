# Evaluation Protocol

Normalize model responses before scoring: strip one leading `<think>...</think>` reasoning block (including an empty Qwen3 block), and when FMAPI content is a list of reasoning/text blocks, score and persist the concatenated text blocks rather than the container representation.

Split roles are fixed: train fits weights or prompts, development chooses strategy and checkpoints, and test is scored once after decisions are frozen. Fingerprint every split and check near duplicates across splits before training.

Report strict task metric, valid-format rate, latency distribution, throughput, token use, and estimated cost. For PHEE, also report main-argument exact match and token F1 for comparison with published sanity ranges. Do not call differing published protocols a leaderboard comparison.

Use the same prompts, decoding policy, input order, and metric implementation for all eligible baselines. Compare frontier zero-shot, frontier few-shot, base small model, optional GEPA, SFT, and any distillation or RL candidate. Match SFT and SFT+RL compute and disclose the matching rule.

The model card must state license, data provenance, intended use, exclusions, schema, metric, baseline table, known errors, hardware, package pins, and the selected UC model version.
