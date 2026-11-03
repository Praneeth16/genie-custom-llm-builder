# LoRA SFT

Start with `Qwen/Qwen3-4B-Instruct-2507`. Use chat-format examples and retain genuine empty outputs. Target attention and MLP projections. Start near rank 32 for small gold datasets; keep alpha fixed when comparing ranks. A LoRA learning rate can be about ten times a full-fine-tune rate, but choose it from the development split.

Run the 30-step probe before a full run. Compare data-size curves when the result affects strategy. Save the adapter and tokenizer to a UC volume. Record exact package versions, data fingerprint, seed, prompt template, and checkpoint.
