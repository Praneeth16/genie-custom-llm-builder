from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from metrics import extraction_metrics
from task_adapter import get_adapter


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompts_for(rows: list[dict], examples: list[dict]) -> list[list[dict[str, str]]]:
    adapter = get_adapter({"adapter": "phee"})
    return [adapter.prompt_messages(row["input_text"], examples) for row in rows]


def local_predictions(base_model: str, adapter: str | None, prompts: list[list[dict[str, str]]]) -> tuple[list[str], float]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    engine = LLM(model=base_model, enable_lora=adapter is not None, max_lora_rank=64)
    request = LoRARequest("task-adapter", 1, adapter) if adapter else None
    started = time.perf_counter()
    outputs = engine.chat(prompts, SamplingParams(temperature=0, max_tokens=512), lora_request=request)
    return [output.outputs[0].text for output in outputs], time.perf_counter() - started


def fmapi_predictions(endpoint: str, prompts: list[list[dict[str, str]]]) -> tuple[list[str], float]:
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    started, values = time.perf_counter(), []
    for messages in prompts:
        response = client.predict(endpoint=endpoint, inputs={"messages": messages, "max_tokens": 512})
        values.append(response["choices"][0]["message"]["content"])
    return values, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-dataset")
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-model")
    source.add_argument("--fmapi-endpoint")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.adapter and not args.base_model:
        parser.error("--adapter requires --base-model")
    if args.few_shot and not args.train_dataset:
        parser.error("--few-shot requires --train-dataset")
    import mlflow
    rows = load_rows(args.dataset)
    train_rows = load_rows(args.train_dataset) if args.train_dataset else []
    examples = random.Random(args.seed).sample(train_rows, args.few_shot)
    prompts = prompts_for(rows, examples)
    predictions, elapsed = fmapi_predictions(args.fmapi_endpoint, prompts) if args.fmapi_endpoint else local_predictions(args.base_model, args.adapter, prompts)
    per_row = [extraction_metrics(prediction, row["gold"], row["input_text"]) for prediction, row in zip(predictions, rows)]
    metrics = {key: sum(item[key] for item in per_row) / len(per_row) for key in per_row[0]}
    metrics["examples_per_second"] = len(rows) / elapsed
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"rows": [{"prediction": prediction, **score} for prediction, score in zip(predictions, per_row)]}, "predictions.json")


if __name__ == "__main__":
    main()
