Use $custom-llm-builder for phase 1 (task card) of a new project in `examples/ade-phee/`. Task: pharmacovigilance adverse drug event extraction from sentences in medical case reports, using PHEE V2. Base model `Qwen/Qwen3-4B-Instruct-2507`. Read `research/03-dataset-phee.md` and `research/04-databricks-facts-and-gotchas.md` for facts.

Do this:
1. Write `examples/ade-phee/task_card.yaml`: task, data source (PHEE V2, `https://github.com/ZhaoyueSun/phee-with-chatgpt`, MIT), output schema, metric (gated strict event F1 as primary; also valid-format rate and PHEE main-argument EM and token F1), quality target (dev strict event F1 at least 0.05 above the best frontier few-shot baseline, to be filled in after baselines), latency and cost targets (batch: at least 20 sentences per second on one H100; cost per 1,000 sentences lower than the frontier model), base model, Databricks profile `fe-vm-lakebase-praneeth`, UC paths below.
2. Clone PHEE V2 into a temp folder outside the repo and inspect the real data layout before converting. Adjust `skill/custom-llm-builder/scripts/prepare_data.py` to the real layout if needed (keep its tests passing, add a test for the real record shape). Decide how to map PHEE arguments to the schema fields and write the mapping into the task card. If some gold events cannot satisfy the schema (for example a missing treatment or a span that is not verbatim in the sentence), report counts by reason in the data card; do not silently drop them.
3. Run data prep to `examples/ade-phee/data/` (gitignored large files are fine; keep the data card and a 20-row sample in git). Check split counts against the paper (about 2.9k / 961 / 968) and near duplicates across splits.
4. With the Databricks CLI and profile `fe-vm-lakebase-praneeth`: create schema `serverless_lakebase_praneeth_catalog.custom_llm` and volume `artifacts` if they do not exist, then upload train/dev/test JSONL and the data card to `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/`. Do not touch any other schema.
5. Hand-check 5 converted dev rows against the source records and note the result in the data card.

Rules: no AI Runtime runs in this session. Do not print tokens. Plain, simple prose.

Reply with: the task card, the data card (counts, overlap, dropped or invalid gold by reason, hand-check result), the UC paths created, and any doubts.
