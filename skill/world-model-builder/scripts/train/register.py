"""Register the trained models in Unity Catalog and point the `champion` alias at them.

python train/register.py --artifacts /Volumes/c/s/artifacts --schema c.s sensor encoder
"""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

import jepa_wm
from jepa_wm.tracking import child_run

PACKAGE = str(Path(jepa_wm.__file__).parent)


def pins(*packages: str) -> list[str]:
    """Pin to the versions this run imported, minus local build tags such as +cu130.

    Read from the modules, not from package metadata: the AI Runtime image ships some of them
    under a different distribution name (mlflow, for one).
    """
    modules = {"pillow": "PIL"}
    return [f"{p}=={importlib.import_module(modules.get(p, p)).__version__.split('+')[0]}" for p in packages]


def register(name: str, model, artifacts: dict, example: pd.DataFrame, output: pd.DataFrame, requirements) -> str:
    info = mlflow.pyfunc.log_model(
        name=name.split(".")[-1],
        python_model=model,
        artifacts=artifacts,
        code_paths=[PACKAGE],
        pip_requirements=requirements,
        signature=infer_signature(example, output),
        input_example=example,
    )
    version = mlflow.register_model(info.model_uri, name).version
    mlflow.MlflowClient().set_registered_model_alias(name, "champion", version)
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=("sensor", "encoder"))
    parser.add_argument("--artifacts", required=True, help="Volume folder holding sensor/ and pusht/")
    parser.add_argument("--schema", required=True, help="catalog.schema to register into")
    args = parser.parse_args()

    mlflow.set_registry_uri("databricks-uc")
    artifacts = Path(args.artifacts)
    versions = {}
    with child_run("register", phase="register"), tempfile.TemporaryDirectory() as scratch:
        if "sensor" in args.models:
            from jepa_wm.serving.sensor_pyfunc import SensorJepaModel

            run = json.loads((artifacts / "sensor/seed_0/results.json").read_text())["args"]
            config = Path(scratch) / "config.json"
            config.write_text(json.dumps({"channels": run["channels"], "window": run["window"]}))
            example = pd.DataFrame({"window": [np.zeros((run["window"], run["channels"])).tolist()]})
            output = pd.DataFrame({"anomaly_score": [0.0], "embedding": [np.zeros(128, dtype=np.float32)]})
            versions["sensor_jepa"] = register(
                f"{args.schema}.sensor_jepa",
                SensorJepaModel(),
                {"weights": str(artifacts / "sensor/seed_0/jepa.pt"), "config": str(config)},
                example,
                output,
                pins("mlflow", "torch", "numpy", "pandas"),
            )
        if "encoder" in args.models:
            from huggingface_hub import snapshot_download

            from jepa_wm.serving.encoder_pyfunc import FrameEncoderModel

            # The weights go into this workspace's registry only; they are never committed.
            used = json.loads((artifacts / "pusht/encoder.json").read_text())
            encoder = snapshot_download(used["repo"], local_dir=Path(scratch) / "encoder")
            grid = Path(scratch) / "grid.json"
            grid.write_text(json.dumps({"grid": used["grid"]}))
            pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
            example = pd.DataFrame({"image": [pixel]})
            output = pd.DataFrame(
                {
                    "embedding": [np.zeros(1024, dtype=np.float32)],
                    "agent_x": [np.float32(0)],
                    "agent_y": [np.float32(0)],
                }
            )
            versions["vjepa2_encoder"] = register(
                f"{args.schema}.vjepa2_encoder",
                FrameEncoderModel(),
                {"encoder": encoder, "config": str(grid), "probe": str(artifacts / "pusht/probe/probe.pt")},
                example,
                output,
                pins("mlflow", "torch", "transformers", "numpy", "pandas", "pillow"),
            )
    print(json.dumps(versions))
    (artifacts / "registration.json").write_text(json.dumps(versions))


if __name__ == "__main__":
    main()
