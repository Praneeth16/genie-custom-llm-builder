from __future__ import annotations

import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/train.jsonl")
DEV = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/data/phee_v2/dev.jsonl")
SFT = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/sft/adapters/full")
ROOT = Path("/Volumes/serverless_lakebase_praneeth_catalog/custom_llm/artifacts/runs/rl")
CODE = Path(os.environ["CODE_SOURCE_PATH"])
SCRIPTS = CODE / "skill/custom-llm-builder/scripts"
EXAMPLE = CODE / "examples/ade-phee"
sys.path.insert(0, str(SCRIPTS))


def write(name: str, value: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


def run(name: str, command: list[str], *, allow_failure: bool = False) -> dict:
    started = time.perf_counter()
    result = subprocess.run(command, text=True)
    path = ROOT / f"{name}.json"
    report = json.loads(path.read_text()) if path.exists() else {"step": name}
    report.setdefault("accelerator_minutes", (time.perf_counter() - started) / 60)
    report["returncode"] = result.returncode
    write(name, report)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    return report


def install() -> None:
    import mlflow
    before = importlib.metadata.version("torch")
    report = {"step": "install_pins", "status": "failed", "torch_before": before}
    started = time.perf_counter()
    try:
        with mlflow.start_run():
            with mlflow.start_run(nested=True, run_name="install_pins", tags={"phase": "rl", "step": "install_pins"}) as active:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "trl==1.15.0", "unsloth==2026.11.1"], text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1500:])
                versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "vllm", "unsloth", "peft")}
                if versions["torch"] != before:
                    raise RuntimeError(f"torch changed from {before} to {versions['torch']}")
                check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
                stack = {"torch", "transformers", "trl", "vllm", "unsloth", "unsloth-zoo", "peft", "accelerate", "datasets", "bitsandbytes", "xformers"}
                conflicts = [line for line in check.stdout.splitlines() if line.split(" ", 1)[0].lower() in stack]
                if conflicts:
                    raise RuntimeError(f"training-stack conflicts: {conflicts}")
                report.update(status="completed", versions=versions, unrelated_pip_conflicts=[line for line in check.stdout.splitlines() if line and line not in conflicts], mlflow_run_id=active.info.run_id)
    finally:
        report["accelerator_minutes"] = (time.perf_counter() - started) / 60
        write("install_pins", report)


def signals(name: str, adapter: Path | None) -> dict:
    command = [sys.executable, str(SCRIPTS / "signals.py"), "--model", MODEL, "--dev-dataset", str(DEV), "--subset-size", "64", "--samples", "8", "--seed", "42", "--output", str(ROOT / f"{name}.json"), "--tag", "phase=rl", "--tag", f"step={name}"]
    if adapter:
        command += ["--adapter", str(adapter)]
    return run(name, command)


def dynamic_prompts() -> dict:
    from metrics import extraction_metrics
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]
    engine = LLM(model=MODEL, enable_lora=True, max_lora_rank=64)
    request = LoRARequest("sft-full", 1, str(SFT))
    kept = []
    batch_size = 128
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        outputs = engine.chat([row["prompt"] for row in batch], SamplingParams(n=8, temperature=1.0, max_tokens=256, seed=42), lora_request=request)
        for row, output in zip(batch, outputs):
            rewards = [extraction_metrics(item.text, row["gold"], row["input_text"])["strict_event_f1"] for item in output.outputs]
            if statistics.pstdev(rewards) > 0:
                kept.append(row)
    destination = ROOT / "datasets/rl_prompts.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row) for row in kept) + "\n")
    report = {"step": "dynamic_sampling", "status": "completed", "train_prompts": len(rows), "surviving_prompts": len(kept), "survival_rate": len(kept) / len(rows)}
    write("dynamic_sampling", report)
    return report


def evaluate(name: str, adapter: Path) -> dict:
    return run(f"evaluate_{name}", [sys.executable, str(SCRIPTS / "evaluate.py"), "--dataset", str(DEV), "--base-model", MODEL, "--adapter", str(adapter), "--output", str(ROOT / f"evaluate_{name}.json"), "--tag", "phase=rl", "--tag", f"step=evaluate_{name}"])


def probe_passes(report: dict) -> bool:
    metrics = report.get("trainer_metrics", {})
    reward_values = [value for key, value in metrics.items() if key == "reward" or key.endswith("/reward") or "rewards/" in key]
    zero_std = [value for key, value in metrics.items() if "frac_reward_zero_std" in key]
    clipped = [value for key, value in metrics.items() if "clipped" in key and "ratio" in key]
    report["probe_checks"] = {
        "reward_nonzero": any(value != 0 for value in reward_values),
        "frac_reward_zero_std": zero_std[-1] if zero_std else None,
        "clipped_ratio": clipped[-1] if clipped else None,
        "seconds_per_step": metrics.get("seconds_per_step"),
        "peak_memory_fraction": report.get("peak_memory_fraction"),
    }
    return report["returncode"] == 0 and report["probe_checks"]["reward_nonzero"] and zero_std and clipped


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    install()
    tuned = signals("signals_sft_full", SFT)
    base = signals("signals_base", None)
    differs = tuned["reward_groups"] != base["reward_groups"]
    gate = tuned["fraction_zero_reward_std"] <= 0.6 and differs
    comparison = {"step": "gate", "status": "passed" if gate else "failed", "fraction_zero_reward_std": tuned["fraction_zero_reward_std"], "tuned_rollouts_differ_from_base": differs, "tuned_mean_reward": tuned["mean_reward"], "base_mean_reward": base["mean_reward"]}
    write("gate", comparison)
    if not gate:
        comparison["recommended_alternative"] = "Continue supervised fine-tuning or use off-policy Claude distillation; GRPO lacks within-prompt reward variance."
        write("gate", comparison)
        evaluate("sft_full", SFT)
        return
    sampled = dynamic_prompts()
    if not sampled["surviving_prompts"]:
        raise RuntimeError("dynamic sampling retained no prompts")
    common = ["--base-model", MODEL, "--adapter", str(SFT), "--dataset", str(ROOT / "datasets/rl_prompts.jsonl")]
    probe_output = ROOT / "adapters/grpo_probe"
    probe = run("grpo_probe", [sys.executable, str(EXAMPLE / "train_grpo_arm.py"), *common, "--output", str(probe_output), "--result-output", str(ROOT / "grpo_probe.json"), "--path", "unsloth", "--max-steps", "10"], allow_failure=True)
    path = "unsloth"
    if not probe_passes(probe):
        write("grpo_probe", probe)
        path = "trl"
        probe = run("grpo_probe_trl", [sys.executable, str(EXAMPLE / "train_grpo_arm.py"), *common, "--output", str(probe_output), "--result-output", str(ROOT / "grpo_probe_trl.json"), "--path", "trl", "--max-steps", "10"])
        if not probe_passes(probe):
            write("grpo_probe_trl", probe)
            raise RuntimeError("both Unsloth and plain TRL GRPO probes failed their learning-signal/telemetry checks")
    write("grpo_path", {"step": "grpo_path", "status": "completed", "path": path, "probe": probe})
    run("sft_plus", [sys.executable, str(EXAMPLE / "train_sft_plus.py"), "--adapter", str(SFT), "--dataset", str(TRAIN), "--output", str(ROOT / "adapters/sft_plus"), "--result-output", str(ROOT / "sft_plus.json"), "--minutes", "40"])
    run("grpo", [sys.executable, str(EXAMPLE / "train_grpo_arm.py"), *common, "--output", str(ROOT / "adapters/grpo"), "--result-output", str(ROOT / "grpo.json"), "--path", path, "--minutes", "40"])
    evaluate("sft_full", SFT)
    evaluate("sft_plus", ROOT / "adapters/sft_plus")
    evaluate("grpo", ROOT / "adapters/grpo")


if __name__ == "__main__":
    main()
