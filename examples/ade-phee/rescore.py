"""Re-score saved dev predictions with the current metric code.

Usage: MLFLOW_TRACKING_URI=databricks DATABRICKS_CONFIG_PROFILE=<profile> \
  python rescore.py <label>=<mlflow_run_id> ...
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "skill" / "custom-llm-builder" / "scripts"))

import mlflow  # noqa: E402
from metrics import REASONING_PREFIX, extraction_metrics, response_text  # noqa: E402

dev = [json.loads(line) for line in (HERE / "data" / "dev.jsonl").read_text().splitlines() if line.strip()]
results = {}
for spec in sys.argv[1:]:
    label, run_id = spec.split("=", 1)
    with tempfile.TemporaryDirectory() as tmp:
        rows = json.loads(Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="predictions.json", dst_path=tmp)).read_text())["rows"]
    assert len(rows) == len(dev), f"{label}: {len(rows)} predictions for {len(dev)} dev rows"
    per_row = [extraction_metrics(row["prediction"], item["gold"], item["input_text"]) for row, item in zip(rows, dev)]
    metrics = {key: round(sum(m[key] for m in per_row) / len(per_row), 4) for key in per_row[0]}
    metrics["reasoning_prefix_rate"] = round(sum(bool(REASONING_PREFIX.match(response_text(r["prediction"]) or "")) for r in rows) / len(rows), 4)
    results[label] = {"run_id": run_id, **metrics}
    print(label, json.dumps(metrics))
(HERE / "rescored_dev.json").write_text(json.dumps(results, indent=2) + "\n")
