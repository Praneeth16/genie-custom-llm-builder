from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rewards import extraction_reward


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def local_predictions(model: str, rows: list[dict]) -> tuple[list[str], float]:
    from vllm import LLM, SamplingParams
    engine = LLM(model=model, enable_lora=True)
    prompts = [row["messages"][0]["content"] for row in rows]
    started = time.perf_counter()
    outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=512))
    return [output.outputs[0].text for output in outputs], time.perf_counter() - started


def fmapi_predictions(endpoint: str, rows: list[dict]) -> tuple[list[str], float]:
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    started = time.perf_counter()
    values = []
    for row in rows:
        response = client.predict(endpoint=endpoint, inputs={"messages": [row["messages"][0]], "max_tokens": 512})
        values.append(response["choices"][0]["message"]["content"])
    return values, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--fmapi-endpoint")
    args = parser.parse_args()
    import mlflow
    rows = load_rows(args.dataset)
    predictions, elapsed = fmapi_predictions(args.fmapi_endpoint, rows) if args.fmapi_endpoint else local_predictions(args.model, rows)
    scores = [extraction_reward(prediction, row["gold"], row["input_text"]) for prediction, row in zip(predictions, rows)]
    with mlflow.start_run():
        mlflow.log_metrics({"strict_event_f1": sum(scores) / len(scores), "examples_per_second": len(rows) / elapsed})
        mlflow.log_dict({"rows": [{"prediction": prediction, "score": score} for prediction, score in zip(predictions, scores)]}, "predictions.json")


if __name__ == "__main__":
    main()
