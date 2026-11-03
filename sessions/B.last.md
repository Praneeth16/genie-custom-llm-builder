**Delivered**
- Task card: `examples/ade-phee/task_card.yaml:1`
- Data card: `examples/ade-phee/data/data_card.md:1`
- Sample: `examples/ade-phee/data/sample-20.jsonl:1`
- Counts: 2,898 train / 961 dev / 968 test; zero cross-split exact or near duplicates.
- Gold audit: 921 incompatible events reported by reason; all sentences retained. Hand-check passed 5/5.
- UC schema: `serverless_lakebase_praneeth_catalog.custom_llm`
- UC volume path: `/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/`
- Doubts: the schema cannot represent missing effects or multiple/discontinuous spans; the numeric quality target remains pending frontier baselines.
- No AI Runtime runs were started.