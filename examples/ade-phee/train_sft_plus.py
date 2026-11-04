from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--minutes", type=float, default=40)
    args = parser.parse_args()

    import mlflow
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    class TimeLimit(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.perf_counter()

        def on_step_end(self, _args, _state, control, **_kwargs):
            if time.perf_counter() - self.started >= args.minutes * 60:
                control.should_training_stop = True

    started = time.perf_counter()
    report = {"step": "sft_plus", "status": "failed", "budget_minutes": args.minutes}
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=2048, load_in_4bit=False)
        data = load_dataset("json", data_files=args.dataset, split="train")
        config = SFTConfig(
            output_dir="/tmp/sft_plus", num_train_epochs=1000, learning_rate=5e-5,
            seed=42, report_to="mlflow", completion_only_loss=True,
            save_strategy="no", per_device_train_batch_size=1,
        )
        with mlflow.start_run():
            with mlflow.start_run(nested=True, run_name="sft_plus", tags={"phase": "rl", "step": "sft_plus"}) as run:
                trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data, args=config, callbacks=[TimeLimit()])
                result = trainer.train()
                model.save_pretrained(args.output)
                tokenizer.save_pretrained(args.output)
                report.update(status="completed", mlflow_run_id=run.info.run_id, optimizer_steps=result.global_step, train_loss=result.training_loss)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["accelerator_minutes"] = (time.perf_counter() - started) / 60
        Path(args.result_output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
