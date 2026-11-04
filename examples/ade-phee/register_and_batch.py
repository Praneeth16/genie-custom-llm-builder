from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from ade_model import ADEExtractor, BASE_MODEL


UC_MODEL = "serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor"
ADAPTER = "/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/adapters/full_v2"
TEST = "/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/test.jsonl"
ROOT = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/phase5")
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "skill" / "custom-llm-builder" / "scripts"


def pinned_requirements() -> list[str]:
    packages = ("torch", "transformers", "vllm", "peft", "pandas", "databricks-sdk")
    return [f"mlflow=={mlflow.__version__}"] + [f"{package}=={importlib.metadata.version(package)}" for package in packages]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_registry_uri("databricks-uc")
    requirements = pinned_requirements()
    (ROOT / "registered_requirements.txt").write_text("\n".join(requirements) + "\n")
    signature = infer_signature(
        [{"sentence": "Aspirin caused nausea."}],
        [json.dumps({"events": []})],
    )
    with mlflow.start_run():
        with mlflow.start_run(nested=True, run_name="register_sft_full_v2", tags={"phase": "test", "step": "register", "role": "final"}) as run:
            info = mlflow.pyfunc.log_model(
                name="ade_extractor",
                python_model=ADEExtractor(),
                artifacts={"adapter": ADAPTER, "prompt_builder": str(SCRIPTS / "task_adapter.py")},
                code_paths=[str(HERE / "ade_model.py"), str(SCRIPTS / "task_adapter.py")],
                pip_requirements=requirements,
                signature=signature,
                input_example={"sentence": "Aspirin caused nausea."},
                metadata={"base_model_id": BASE_MODEL, "adapter": "sft-full-v2", "task": "ade_extraction"},
            )
            version = mlflow.register_model(info.model_uri, UC_MODEL, env_pack="databricks_model_serving")
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(UC_MODEL, "champion", version.version)
            mlflow.log_params({"uc_model": UC_MODEL, "uc_version": version.version, "alias": "champion"})
            registration = {"model": UC_MODEL, "version": version.version, "alias": "champion", "run_id": run.info.run_id}
            (ROOT / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    subprocess.run([
        sys.executable,
        str(HERE / "batch_score.py"),
        "--input-jsonl", TEST,
        "--limit", "50",
        "--model-uri", f"models:/{UC_MODEL}@champion",
        "--warehouse-id", os.environ["WAREHOUSE_ID"],
        "--run-key", "ade-phee-phase5-test50-v1",
        "--report-output", str(ROOT / "batch_score_50.json"),
    ], check=True)


if __name__ == "__main__":
    main()
