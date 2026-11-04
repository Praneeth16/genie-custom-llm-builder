Use $custom-llm-builder for phase 5 (evaluate and ship) of `examples/ade-phee/`. Profile `fe-vm-lakebase-praneeth`.

## Decision freeze (project lead, before any test scoring)
GRPO (0.7818) and SFT-full v2 (0.7817) are tied on dev: the gap is 0.0001, far below the 0.008 noise level. A dev tie goes to the cheaper model, so the final candidate is **SFT-full v2** (`.../runs/sft/adapters/full_v2/`). GRPO is reported on test as a pre-declared secondary result and is not used for selection. Write this freeze into `strategy_decision.md` with a timestamp before you score test.

## 1. Test split, scored once (968 rows, same prompt and decoding as dev)
- Locally via FMAPI: Claude Sonnet 5.5 zero-shot and 5-shot (same seed-42 train examples). Concurrency 8.
- One AIR H100 workload: base Qwen3-4B 5-shot, SFT-full v2 (final), GRPO arm B (secondary). Use distinct run names. Record throughput.
- Tag every run `phase=test`. Add a "Test results" table to `strategy_decision.md`: model, role (baseline / final / secondary), strict event F1, valid rate, PHEE EM, PHEE token F1, precision, recall, throughput, tokens.

## 2. Register the final model in Unity Catalog
Register SFT-full v2 as `serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor`: an MLflow pyfunc that loads the base model plus the LoRA adapter with vLLM and returns the JSON string, with the base model id, adapter files, prompt builder and pinned requirements. Reuse the pattern in `02-assets/databricks-agent-loop/train/train.py` (`mlflow.set_registry_uri("databricks-uc")`, `register_model(..., env_pack="databricks_model_serving")`) if it fits. Do this inside the same AIR workload as step 1 so it runs on a GPU. Set the alias `champion` on the new version.

## 3. Batch scoring job
Write `examples/ade-phee/batch_score.py` and a workload YAML that reads a Delta table or JSONL of sentences, loads the `@champion` model version, and writes extracted events to a Delta table `serverless_lakebase_praneeth_catalog.custom_llm.ade_events`. Run it once on 50 test rows inside the same workload and check that every output parses. Then use `databricks air convert-to-dabs` to produce a bundle under `examples/ade-phee/bundle/` with a weekly schedule set to PAUSED. Validate the bundle; deploy it only if validation passes, and leave the schedule paused.

## 4. Model card
`examples/ade-phee/MODEL_CARD.md`: license (PHEE MIT, Qwen Apache-2.0), data provenance and source commit, intended use (research and demo pharmacovigilance triage, not clinical decisions), exclusions, schema, metric, dev and test tables, known errors (show 5 real test errors with short notes), hardware, package versions, training recipe and the strategy path (signals, SFT, chat-template bug, gate, GRPO result), the UC model version and alias.

Rules: dry-run, idempotency key, `max_retries: 0`, short log snapshots only, never cancel runs you did not start. If a browser login is requested, stop and say so. Reply with the test table, UC model name and version, batch job result (rows written, parse rate), bundle status, and links.
