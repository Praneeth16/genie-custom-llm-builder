from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path
from typing import Any

from metrics import extraction_metrics
from task_adapter import get_adapter


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompts_for(rows: list[dict], examples: list[dict]) -> list[list[dict[str, str]]]:
    adapter = get_adapter({"adapter": "phee"})
    return [adapter.prompt_messages(row["input_text"], examples) for row in rows]


def local_predictions(base_model: str, adapter: str | None, prompts: list[list[dict[str, str]]]) -> tuple[list[str], float, dict[str, int]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    engine = LLM(model=base_model, enable_lora=adapter is not None, max_lora_rank=64)
    request = LoRARequest("task-adapter", 1, adapter) if adapter else None
    started = time.perf_counter()
    outputs = engine.chat(prompts, SamplingParams(temperature=0, max_tokens=512), lora_request=request)
    return [output.outputs[0].text for output in outputs], time.perf_counter() - started, {}


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    return status == 429 or status is not None and 500 <= status < 600


def _fmapi_prediction(client: Any, endpoint: str, messages: list[dict[str, str]], max_attempts: int = 6) -> tuple[str, dict[str, int]]:
    for attempt in range(max_attempts):
        try:
            response = client.predict(endpoint=endpoint, inputs={"messages": messages, "max_tokens": 512})
            usage = response.get("usage") or {}
            return response["choices"][0]["message"]["content"], {
                key: int(usage[key]) for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in usage
            }
        except Exception as error:
            if attempt == max_attempts - 1 or not _is_retryable(error):
                raise
            time.sleep(min(2 ** attempt, 16))
    raise AssertionError("unreachable")


def fmapi_predictions(
    endpoint: str,
    prompts: list[list[dict[str, str]]],
    concurrency: int = 8,
    client: Any | None = None,
) -> tuple[list[str], float, dict[str, int]]:
    if client is None:
        from mlflow.deployments import get_deploy_client
        client = get_deploy_client("databricks")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda messages: _fmapi_prediction(client, endpoint, messages), prompts))
    usage = {
        key: sum(item.get(key, 0) for _, item in results)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(key in item for _, item in results)
    }
    return [prediction for prediction, _ in results], time.perf_counter() - started, usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-dataset")
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--experiment-name")
    parser.add_argument("--output")
    parser.add_argument("--tag", action="append", default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-model")
    source.add_argument("--fmapi-endpoint")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.adapter and not args.base_model:
        parser.error("--adapter requires --base-model")
    if args.few_shot and not args.train_dataset:
        parser.error("--few-shot requires --train-dataset")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    import mlflow
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    rows = load_rows(args.dataset)
    train_rows = load_rows(args.train_dataset) if args.train_dataset else []
    examples = random.Random(args.seed).sample(train_rows, args.few_shot)
    prompts = prompts_for(rows, examples)
    predictions, elapsed, usage = fmapi_predictions(args.fmapi_endpoint, prompts, args.concurrency) if args.fmapi_endpoint else local_predictions(args.base_model, args.adapter, prompts)
    per_row = [extraction_metrics(prediction, row["gold"], row["input_text"]) for prediction, row in zip(predictions, rows)]
    metrics = {key: sum(item[key] for item in per_row) / len(per_row) for key in per_row[0]}
    metrics["examples_per_second"] = len(rows) / elapsed
    metrics.update({f"usage_{key}": value for key, value in usage.items()})
    tags = dict(tag.split("=", 1) for tag in args.tag)
    report = {"metrics": metrics, "usage": usage, "elapsed_seconds": elapsed, "rows": len(rows)}
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_metrics(metrics)
        if args.experiment_name:
            mlflow.log_params({"model": args.fmapi_endpoint or args.base_model, "shots": args.few_shot, "seed": args.seed})
        mlflow.log_dict({"rows": [{"prediction": prediction, **score} for prediction, score in zip(predictions, per_row)]}, "predictions.json")
        report["mlflow_run_id"] = run.info.run_id
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
