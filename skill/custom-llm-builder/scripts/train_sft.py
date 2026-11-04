from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--output", default=os.environ.get("ADAPTER_OUTPUT"))
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--result-output")
    parser.add_argument("--step-name", default="sft")
    parser.add_argument("--phase", default="train")
    args = parser.parse_args()
    if not args.output:
        parser.error("--output or ADAPTER_OUTPUT is required")
    import mlflow
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from chat_template import use_reference_chat_template
    from training_callback import build_probe_callback

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=2048, load_in_4bit=False)
    use_reference_chat_template(tokenizer, args.model)
    model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
    data = load_dataset("json", data_files={"train": args.train_dataset, "dev": args.dev_dataset})
    first_row = data["train"][0]
    rendered_training_example = tokenizer.apply_chat_template(
        first_row["prompt"] + first_row["completion"], tokenize=False
    )
    config = SFTConfig(output_dir=f"/tmp/{args.step_name}", max_steps=args.max_steps, num_train_epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, report_to="mlflow", completion_only_loss=True, eval_strategy="no", save_strategy="no")
    callback = build_probe_callback()
    report = {"step": args.step_name, "status": "failed"}
    started = time.perf_counter()
    try:
        with mlflow.start_run():
            with mlflow.start_run(nested=True, run_name=args.step_name, tags={"phase": args.phase, "step": args.step_name, "method": "lora_sft", "model": args.model}) as run:
                trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data["train"], eval_dataset=None, args=config, callbacks=[callback])
                result = trainer.train()
                model.save_pretrained(args.output)
                tokenizer.save_pretrained(args.output)
                elapsed = time.perf_counter() - started
                report.update({"status": "completed", "mlflow_run_id": run.info.run_id, "train_rows": len(data["train"]), "rendered_training_example": rendered_training_example, "optimizer_steps": result.global_step, "train_loss": result.training_loss, "seconds_per_step": elapsed / result.global_step, "peak_memory_fraction": callback.peak_memory_fraction, "accelerator_minutes": elapsed / 60})
                mlflow.log_metrics({key: value for key, value in report.items() if isinstance(value, (int, float))})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if args.result_output:
            Path(args.result_output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
