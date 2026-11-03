**File Tree**
```text
skill/custom-llm-builder/
├── SKILL.md
├── agents/openai.yaml
├── references/{air-glue,eval-protocol,rewards,strategy-selection}.md
├── references/recipes/{sft-lora,distill-offpolicy,gkd-onpolicy,grpo-dapo,gepa-baseline}.md
├── scripts/{task_adapter,rewards,prepare_data,train_sft,train_grpo,evaluate,probe}.py
├── scripts/workloads/{probe,sft,grpo,eval}.yaml
└── tests/{conftest,test_rewards,test_task_adapter}.py
```

**Tests**
- `13 passed in 0.03s`
- Skill validation: `Skill is valid!`

**Open Doubts**
- PHEE V2’s exact upstream JSON layout is not documented in the research; `prepare_data.py` handles common field variants but should be verified after download.
- Installed `databricks_ai_v6` vLLM compatibility remains unverified and must be checked during the probe.