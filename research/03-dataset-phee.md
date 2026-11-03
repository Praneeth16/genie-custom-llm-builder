# Dataset: PHEE for adverse drug event extraction (research as of 2026-11-04)

## Choice
PHEE (Pharmacovigilance Event Extraction). It is the only candidate that is openly licensed, has span-level role labels, and has published baselines.

- License: MIT for both repos, checked with the GitHub API on 2026-11-04: [ZhaoyueSun/PHEE](https://github.com/ZhaoyueSun/PHEE) (V1, data under `data/{clean,json,eeqa,gen_qa,ace}`) and [ZhaoyueSun/phee-with-chatgpt](https://github.com/ZhaoyueSun/phee-with-chatgpt) (V2.0, revised annotations, data under `data/`).
- HF port: `sarus-tech/phee` (no license on the card; use the GitHub source as the authority).
- Size: 4,827 sentences from medical case reports. Split 2.9k train / 961 dev / 968 test.
- Labels: events of type `Adverse_event` or `Potential_therapeutic_event`, with Trigger, Subject, Treatment and Effect spans plus character offsets, and sub-arguments (for example `subject.disorder`, `treatment.drug`, `time_elapsed`, `duration`).
- V2.0 changes: automatic fix for `subject.disorder`, and manual correction of `time_elapsed` and `duration`.

## Published targets (protocols differ, so use as a sanity range, not a leaderboard)
- PHEE paper ([Sun et al. 2022](https://aclanthology.org/2022.emnlp-main.376.pdf)), fine-tuned SciFive generative QA, main arguments: EM 68.60, token F1 80.04. Sub-arguments EM 76.16. BioBERT extractive QA EM 65.70.
- ChatGPT study on V2 ([EACL 2024](https://aclanthology.org/2024.eacl-short.30.pdf)), GPT-3.5 5-shot with BM25 example retrieval: main EM 60.39, token F1 76.15. Zero-shot EM 34.80.
- No published GRPO result on ADE extraction.

## Rejected
- n2c2 2018: data use agreement forbids sharing; blocks a shared workspace and a public repo.
- ade_corpus_v2: license "unknown" on HF. Internal experiments only.
- CADEC: custom CSIRO license. FAERS derivatives: structured tables, no text spans.

## Task framing (spec for the skill's worked example)
Input: one sentence. Output JSON:
```json
{"events": [{"event_type": "Adverse_event", "trigger": "...", "subject": "..." , "treatment": "...", "effect": "..."}]}
```
`subject` may be null. Every string must be copied verbatim from the input.

Reward (also the dev metric):
- 0 if the output is not valid JSON, does not match the schema, or any string is not a substring of the input.
- Otherwise F1 over a one-to-one matching of predicted to gold events. Per-event score is token F1 on treatment and effect (plus trigger), with a small bonus for exact matches.
- Both empty = 1.0. Predicted empty when gold is not empty = 0.0.
- No standalone format bonus, because an empty JSON would earn it.
- Watch for these hacks: empty outputs, copying the whole sentence into every field, many duplicate events (the precision term handles this).

Final evaluation adds the PHEE paper's EM and token F1 on main arguments so the numbers can be compared to the published range.
