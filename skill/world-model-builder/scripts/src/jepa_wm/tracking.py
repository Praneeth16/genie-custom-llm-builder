"""MLflow logging that works inside and outside AI Runtime."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager


@contextmanager
def child_run(name: str, **tags: str):
    """Yield `log(values, step=None)`.

    AI Runtime sets MLFLOW_RUN_ID for the workload's run; `start_run()` attaches to it and each
    step gets a nested child run. Without it (local runs) metrics are only printed.
    """
    active = bool(os.environ.get("MLFLOW_RUN_ID"))

    def log(values: dict, step: int | None = None) -> None:
        print(json.dumps({"step": step, **values}), flush=True)
        if active:
            mlflow.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    if not active:
        yield log
        return
    import mlflow

    with mlflow.start_run(), mlflow.start_run(nested=True, run_name=name, tags=tags):
        yield log
