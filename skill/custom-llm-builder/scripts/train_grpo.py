from __future__ import annotations

import argparse
import os


os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=os.environ.get("ADAPTER_OUTPUT"))
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-generations", type=int, default=4)
    args = parser.parse_args()
    if not args.output:
        parser.error("--output or ADAPTER_OUTPUT is required")
    import mlflow
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    from rewards import gated_extraction_reward

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=2048, fast_inference=True)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    config = GRPOConfig(output_dir="/tmp/grpo", max_steps=args.max_steps, report_to="mlflow", loss_type="dapo", beta=0.0, epsilon=0.2, epsilon_high=0.28, num_generations=args.num_generations, max_completion_length=512, per_device_train_batch_size=1, gradient_accumulation_steps=4, use_vllm=True, vllm_mode="colocate")
    peft = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    with mlflow.start_run():
        mlflow.set_tags({"method": "grpo_dapo", "model": args.model})
        trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[gated_extraction_reward], train_dataset=dataset, args=config, peft_config=peft)
        trainer.train()
        trainer.save_model(args.output)


if __name__ == "__main__":
    main()
