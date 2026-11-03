# Off-Policy Distillation

Use this when a frontier teacher can produce targets but cannot return token logprobs. Generate teacher outputs only for training data. Validate and filter them with the task schema and metric invariants, then train them as SFT targets.

Keep gold and synthetic provenance separate. Report results for gold-only SFT and augmented SFT when gold exists. Do not describe synthetic labels as ground truth. Databricks Claude 5.x FMAPI endpoints accept no `temperature` parameter and provide no logprobs.

Use `scripts/label_with_teacher.py` to create prompt/completion JSONL. It sends no temperature, rejects invalid or non-verbatim outputs, and reports written and rejected counts. Claude 5.x FMAPI endpoints cannot support GKD because they return no logprobs.
