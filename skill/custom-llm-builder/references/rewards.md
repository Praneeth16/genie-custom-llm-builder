# Rewards

The reward must be the task's verifiable metric or a faithful dense form of it. Keep schema validation as a gate, not a separate bonus. A valid empty output must not earn points unless the gold output is also empty.

For extraction, reject invalid JSON, extra or missing schema fields, invalid enum values, and spans that are not copied verbatim from the input. Then compute a one-to-one maximum matching between predicted and gold events. Soft event similarity can average token F1 for required spans with a small exact-match component. Convert the matched total to set precision and recall so duplicate predictions reduce precision.

Test reward hacks: whole-input copying, duplicate events, fabricated spans, empty predictions, and partial spans. For classification, apply the same pattern: strict label schema first, then the frozen classification metric.
