from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--path", choices=("unsloth", "trl"), required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    args = parser.parse_args()

    import mlflow
    import torch
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from rewards import gated_extraction_reward

    class Monitor(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.perf_counter()
            self.metrics: dict[str, float] = {}
            self.peak_memory_fraction = 0.0

        def on_step_begin(self, _args, _state, _control, **_kwargs):
            self.step_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, _args, state, control, **_kwargs):
            elapsed = time.perf_counter() - self.step_started
            peak = torch.cuda.max_memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            self.metrics["seconds_per_step"] = elapsed
            self.peak_memory_fraction = max(self.peak_memory_fraction, peak / total)
            mlflow.log_metrics({"seconds_per_step": elapsed, "peak_memory_fraction": peak / total}, step=state.global_step)
            if args.max_steps < 0 and time.perf_counter() - self.started >= args.minutes * 60:
                control.should_training_stop = True

        def on_log(self, _args, _state, _control, logs=None, **_kwargs):
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)):
                    self.metrics[key] = float(value)

    started = time.perf_counter()
    report = {"step": "grpo_probe" if args.max_steps > 0 else "grpo", "status": "failed", "path": args.path, "budget_minutes": args.minutes}
    monitor = Monitor()
    try:
        if args.path == "unsloth":
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=2048, fast_inference=True)
        else:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
            tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        data = load_dataset("json", data_files=args.dataset, split="train")
        config = GRPOConfig(
            output_dir=f"/tmp/{report['step']}_{args.path}", max_steps=args.max_steps,
            num_train_epochs=1000, learning_rate=5e-6, seed=42, report_to="mlflow",
            loss_type="dapo", beta=0.0, epsilon=0.2, epsilon_high=0.28,
            num_generations=8, max_completion_length=256, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, use_vllm=True, vllm_mode="colocate",
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        )
        run_name = f"{report['step']}_{args.path}"
        with mlflow.start_run():
            with mlflow.start_run(nested=True, run_name=run_name, tags={"phase": "rl", "step": report["step"], "grpo_path": args.path}) as run:
                trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[gated_extraction_reward], train_dataset=data, args=config, callbacks=[monitor])
                result = trainer.train()
                trainer.save_model(args.output)
                tokenizer.save_pretrained(args.output)
                report.update(status="completed", mlflow_run_id=run.info.run_id, optimizer_steps=result.global_step, train_loss=result.training_loss, peak_memory_fraction=monitor.peak_memory_fraction, trainer_metrics=monitor.metrics)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["accelerator_minutes"] = (time.perf_counter() - started) / 60
        Path(args.result_output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
