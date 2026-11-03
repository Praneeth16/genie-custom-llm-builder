from __future__ import annotations

import argparse
import json
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    if not any(part == "--max-steps" or part.startswith("--max-steps=") for part in args.command):
        args.command.extend(["--max-steps", "30"])
    result = subprocess.run(args.command, check=False)
    metrics: dict[str, float | None] = {}
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id:
        from mlflow import MlflowClient
        client = MlflowClient()
        for name in ("probe_peak_gpu_memory_bytes", "probe_seconds_per_step", "completions/clipped_ratio", "frac_reward_zero_std"):
            history = client.get_metric_history(run_id, name)
            metrics[name] = max((point.value for point in history), default=None)
    print(json.dumps({"returncode": result.returncode, "steps": 30, "metrics": metrics}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
