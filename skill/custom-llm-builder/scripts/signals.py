from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from metrics import extraction_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import mlflow
    from vllm import LLM, SamplingParams
    rows = [json.loads(line) for line in Path(args.dev_dataset).read_text().splitlines() if line.strip()][:args.subset_size]
    engine = LLM(model=args.model)
    outputs = engine.chat([row["prompt"] for row in rows], SamplingParams(n=args.samples, temperature=1.0, max_tokens=512, seed=args.seed))
    reward_groups, all_metrics = [], []
    for row, output in zip(rows, outputs):
        group = [extraction_metrics(item.text, row["gold"], row["input_text"]) for item in output.outputs]
        reward_groups.append([item["strict_event_f1"] for item in group])
        all_metrics.extend(group)
    report = {
        "valid_format_rate": statistics.mean(item["valid_format"] for item in all_metrics),
        "mean_reward": statistics.mean(item["strict_event_f1"] for item in all_metrics),
        "mean_within_prompt_reward_std": statistics.mean(statistics.pstdev(group) for group in reward_groups),
        "fraction_zero_reward_std": statistics.mean(statistics.pstdev(group) == 0 for group in reward_groups),
        "empty_output_rate": statistics.mean(item["empty_output"] for item in all_metrics),
    }
    with mlflow.start_run():
        mlflow.log_metrics(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
