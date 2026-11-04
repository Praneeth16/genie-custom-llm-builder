from __future__ import annotations


def use_reference_chat_template(tokenizer, model_id: str) -> None:
    """Force the upstream Hugging Face chat template onto a (possibly Unsloth-mirrored) tokenizer.

    Unsloth loads `Qwen/Qwen3-4B-Instruct-2507` from `unsloth/Qwen3-4B-Instruct-2507`, whose template adds an
    empty `<think></think>` block to every assistant turn. SFT then learns that special-token preamble, which
    greedy decoding hides and sampling breaks (it swaps in `<tool_call>` and similar tokens).
    """
    from transformers import AutoTokenizer

    reference = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = reference.chat_template
    probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    if rendered != reference.apply_chat_template(probe, tokenize=False):
        raise RuntimeError(f"training chat template does not match {model_id}: {rendered!r}")
