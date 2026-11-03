from __future__ import annotations

import json
import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/train.jsonl")
DEV = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/dev.jsonl")
ROOT = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft")
SCRIPTS = Path(os.environ["CODE_SOURCE_PATH"]) / "skill/custom-llm-builder/scripts"


def write_result(step: str, report: dict) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{step}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def run_step(step: str, command: list[str]) -> dict:
    started = time.perf_counter()
    result = subprocess.run(command, text=True)
    elapsed = time.perf_counter() - started
    path = ROOT / f"{step}.json"
    report = json.loads(path.read_text()) if path.exists() else {"step": step}
    report["accelerator_minutes"] = elapsed / 60
    report["returncode"] = result.returncode
    write_result(step, report)
    if result.returncode:
        raise RuntimeError(f"{step} failed with exit code {result.returncode}")
    return report


def install_pins() -> None:
    import mlflow

    step = "install_pins"
    started = time.perf_counter()
    try:
        before_torch = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        before_torch = None
    report = {"step": step, "status": "failed", "torch_before": before_torch}
    try:
        with mlflow.start_run():
            with mlflow.start_run(nested=True, run_name=step, tags={"phase": "train", "step": step}) as run:
                install = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "trl==1.15.0", "unsloth==2026.11.1"],
                    text=True,
                    capture_output=True,
                )
                report["pip_stderr_tail"] = install.stderr[-2000:]
                if install.returncode:
                    raise RuntimeError(f"pinned install failed: {install.stderr[-1000:]}")
                distributions = {"torch": "torch", "transformers": "transformers", "trl": "trl", "vllm": "vllm", "unsloth": "unsloth", "peft": "peft"}
                report["versions"] = {name: importlib.metadata.version(distribution) for name, distribution in distributions.items()}
                if before_torch is not None and report["versions"]["torch"] != before_torch:
                    raise RuntimeError(f"pinned install changed image torch from {before_torch} to {report['versions']['torch']}")
                check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
                stack = {"torch", "transformers", "trl", "vllm", "unsloth", "unsloth-zoo", "unsloth_zoo", "peft", "accelerate", "datasets", "bitsandbytes", "xformers"}
                conflicts = [line for line in check.stdout.splitlines() if line.split(" ", 1)[0].lower() in stack]
                report["unrelated_pip_conflicts"] = [line for line in check.stdout.splitlines() if line and line not in conflicts]
                if conflicts:
                    raise RuntimeError(f"pinned packages conflict with the training stack: {conflicts}")
                report.update({"status": "completed", "mlflow_run_id": run.info.run_id})
                mlflow.log_dict(report["versions"], "versions.json")
                print(json.dumps(report["versions"], indent=2))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["accelerator_minutes"] = (time.perf_counter() - started) / 60
        write_result(step, report)


def subset(size: int) -> Path:
    destination = ROOT / "datasets" / f"train_{size}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = TRAIN.read_text().splitlines()[:size]
    destination.write_text("\n".join(rows) + "\n")
    return destination


def train(step: str, dataset: Path, adapter: Path, *, max_steps: int = -1) -> dict:
    command = [
        sys.executable, str(SCRIPTS / "train_sft.py"), "--train-dataset", str(dataset),
        "--dev-dataset", str(DEV), "--output", str(adapter), "--epochs", "3",
        "--learning-rate", "2e-4", "--seed", "42", "--lora-rank", "32",
        "--max-steps", str(max_steps), "--step-name", step, "--result-output", str(ROOT / f"{step}.json"),
    ]
    return run_step(step, command)


def evaluate(size: str, adapter: Path) -> dict:
    step = f"evaluate_{size}"
    return run_step(step, [
        sys.executable, str(SCRIPTS / "evaluate.py"), "--dataset", str(DEV),
        "--base-model", MODEL, "--adapter", str(adapter), "--output", str(ROOT / f"{step}.json"),
        "--tag", "phase=train", "--tag", f"step={step}",
    ])


def signals(step: str, adapter: Path | None) -> dict:
    command = [
        sys.executable, str(SCRIPTS / "signals.py"), "--model", MODEL, "--dev-dataset", str(DEV),
        "--subset-size", "64", "--samples", "8", "--seed", "42", "--output", str(ROOT / f"{step}.json"),
        "--tag", "phase=train", "--tag", f"step={step}",
    ]
    if adapter:
        command.extend(["--adapter", str(adapter)])
    return run_step(step, command)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    install_pins()
    train("probe", TRAIN, ROOT / "adapters/probe", max_steps=30)
    adapters: dict[str, Path] = {}
    evaluations: dict[str, dict] = {}
    for size, dataset in (("500", subset(500)), ("2000", subset(2000)), ("full", TRAIN)):
        adapters[size] = ROOT / f"adapters/{size}"
        train(f"train_{size}", dataset, adapters[size])
        evaluations[size] = evaluate(size, adapters[size])
    best = max(evaluations, key=lambda size: evaluations[size]["metrics"]["strict_event_f1"])
    tuned = signals("signals_tuned", adapters[best])
    base = signals("signals_base", None)
    comparison = {
        "step": "signals_comparison",
        "status": "completed",
        "best_adapter": best,
        "tuned_rollouts_differ_from_base": tuned["reward_groups"] != base["reward_groups"],
        "tuned_mean_reward": tuned["mean_reward"],
        "base_mean_reward": base["mean_reward"],
        "rl_signal_gate_passes": tuned["fraction_zero_reward_std"] <= 0.6 and tuned["reward_groups"] != base["reward_groups"],
    }
    write_result("signals_comparison", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
