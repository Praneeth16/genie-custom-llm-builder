**Dev Baselines**

| Model | Shots | Strict F1 | Valid | PHEE EM | PHEE token F1 | Empty | Throughput | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5.5 | 0 | 0.6209 | 0.9053 | 0.4657 | 0.6270 | 0.0094 | 2.58/s | 468,843 |
| Claude Sonnet 5.5 | 5 | **0.7112** | 0.9792 | 0.5684 | 0.7192 | 0.0271 | 2.74/s | 1,103,833 |
| Qwen3-4B | 0 | 0.1760 | 0.7055 | 0.1067 | 0.1823 | 0.2695 | 206.17/s | unavailable |
| Qwen3-4B | 5 | 0.3007 | 0.5536 | 0.2241 | 0.3072 | 0.0104 | 178.27/s | unavailable |

List-price documentation was not available, so cost was not estimated.

**Signals / Decision**
- Signals: valid `0.7598`, mean reward `0.2181`, within-prompt std `0.0330`, zero-variance groups `0.75`, empty outputs `0.2754`.
- Decision: LoRA SFT first. The `0.75 > 0.6` zero-variance rule indicates insufficient GRPO signal.
- Target updated to `0.761164` in `examples/ade-phee/task_card.yaml:72`.
- Full rationale and matched-compute plan: `examples/ade-phee/strategy_decision.md:1`.
- Successful AIR run: `267141833486129`.
- Tests: `9 passed`.

**MLflow**
- [Claude zero-shot](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329213/runs/3d75653dc14f4b22aff1a1f036f960c8)
- [Claude five-shot](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329213/runs/a2d1c65482db415db57b71922c6a2c4c)
- [AIR baselines/signals](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329213/runs/432fb7540aa243ac94eeec9e40a25705)
- [Strategy decision](https://fevm-serverless-lakebase-praneeth.cloud.databricks.com/ml/experiments/1438247915329213/runs/3ca94e0a1e904f7eace310371a02a590)

**Bugs Encountered**
- Genie sandbox blocked SDK access to the OS keyring; required a short-lived CLI token with explicit auth type.
- AIR `"6"` selected root Python instead of the AI environment; required `databricks_ai_v6`.
- AIR’s FlashInfer sampler lacked `curand.h`; required `VLLM_USE_FLASHINFER_SAMPLER=0`.
- AIR’s shared MLflow run rejected changing immutable `shots` parameters; AIR evaluations now use tags only.
- No test-split scoring was performed.