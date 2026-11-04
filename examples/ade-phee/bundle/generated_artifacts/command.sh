export INPUT_JSONL=/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/test.jsonl
export VLLM_USE_FLASHINFER_SAMPLER=0
export WAREHOUSE_ID=4d39ac2e32b72a3a

python "$CODE_SOURCE_PATH/batch_score.py" \
  --input-jsonl "$INPUT_JSONL" \
  --output-table serverless_lakebase_praneeth_catalog.custom_llm.ade_events \
  --model-uri models:/serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor@champion \
  --warehouse-id "$WAREHOUSE_ID" \
  --run-key weekly-champion
