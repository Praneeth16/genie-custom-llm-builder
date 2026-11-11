"""Batch-score new sensor windows with the registered champion and write a Delta table.

python train/batch_score.py --schema c.s --volume /Volumes/c/s/artifacts --warehouse <id>
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from jepa_wm.data.synthetic_sensor import generate, windows
from jepa_wm.tracking import child_run
from jepa_wm.uc import table_from_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--steps", type=int, default=50_000, help="length of the new telemetry to score")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    mlflow.set_registry_uri("databricks-uc")
    with child_run("batch_score_sensor", phase="batch_score") as log:
        model = mlflow.pyfunc.load_model(f"models:/{args.schema}.sensor_jepa@champion")
        # Stand-in for new production telemetry. Replace with a read of your own table.
        x, labelled_anomaly = windows(*generate(args.steps, anomaly_fraction=0.05, seed=args.seed), stride=64)
        scored = model.predict(pd.DataFrame({"window": [w.tolist() for w in x]}))
        table = pd.DataFrame(
            {
                "window_id": range(len(x)),
                "start_step": [i * 64 for i in range(len(x))],
                "anomaly_score": scored["anomaly_score"],
                "labelled_anomaly": labelled_anomaly,
            }
        )
        log({"windows_scored": len(table), "mean_score": float(table["anomaly_score"].mean())})

        target = Path(args.volume) / "batch/sensor_scores.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            table.to_parquet(Path(scratch) / "scores.parquet", index=False)
            shutil.copy(Path(scratch) / "scores.parquet", target)
        table_from_parquet(f"{args.schema}.sensor_anomaly_scores", str(target), args.warehouse)


if __name__ == "__main__":
    main()
