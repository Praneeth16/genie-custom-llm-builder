from __future__ import annotations

import argparse
import os


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
    args = parser.parse_args()
    if not args.output:
        parser.error("--output or ADAPTER_OUTPUT is required")
    import mlflow
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from training_callback import build_probe_callback

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=2048, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
    data = load_dataset("json", data_files={"train": args.train_dataset, "dev": args.dev_dataset})
    config = SFTConfig(output_dir="/tmp/sft", max_steps=args.max_steps, num_train_epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, report_to="mlflow", completion_only_loss=True, eval_strategy="epoch")
    with mlflow.start_run():
        mlflow.set_tags({"method": "lora_sft", "model": args.model})
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data["train"], eval_dataset=data["dev"], args=config, callbacks=[build_probe_callback()])
        trainer.train()
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
