from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from metrics import extraction_metrics
from task_adapter import get_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--keep-samples", type=int, default=2, help="sample texts saved per prompt for inspection")
    parser.add_argument("--output")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()
    import mlflow
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in Path(args.dev_dataset).read_text().splitlines() if line.strip()][:args.subset_size]
    adapter = get_adapter({"adapter": "phee"})
    prompts = [adapter.prompt_messages(row["input_text"]) for row in rows]
    engine = LLM(model=args.model, enable_lora=args.adapter is not None, max_lora_rank=64)
    request = LoRARequest("task-adapter", 1, args.adapter) if args.adapter else None
    outputs = engine.chat(prompts, SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=512, seed=args.seed), lora_request=request)
    reward_groups, all_metrics, samples = [], [], []
    for row, output in zip(rows, outputs):
        samples.append([item.text for item in output.outputs[:args.keep_samples]])
        group = [extraction_metrics(item.text, row["gold"], row["input_text"]) for item in output.outputs]
        reward_groups.append([item["strict_event_f1"] for item in group])
        all_metrics.extend(group)
    report = {
        "valid_format_rate": statistics.mean(item["valid_format"] for item in all_metrics),
        "mean_reward": statistics.mean(item["strict_event_f1"] for item in all_metrics),
        "mean_within_prompt_reward_std": statistics.mean(statistics.pstdev(group) for group in reward_groups),
        "fraction_zero_reward_std": statistics.mean(statistics.pstdev(group) == 0 for group in reward_groups),
        "empty_output_rate": statistics.mean(item["empty_output"] for item in all_metrics),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "reward_groups": reward_groups,
        "samples": samples,
    }
    tags = dict(tag.split("=", 1) for tag in args.tag)
    with mlflow.start_run():
        with mlflow.start_run(nested=True, run_name=tags.get("step", "signals"), tags=tags) as run:
            mlflow.log_metrics({key: value for key, value in report.items() if isinstance(value, (int, float))})
            report["mlflow_run_id"] = run.info.run_id
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
