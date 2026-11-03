from __future__ import annotations

from typing import Any

from metrics import extraction_metrics


def extraction_reward(prediction: Any, gold: Any, input_text: str) -> float:
    return extraction_metrics(prediction, gold, input_text)["strict_event_f1"]


def gated_extraction_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    golds = kwargs.get("gold") or kwargs.get("answer") or kwargs.get("targets")
    inputs = kwargs.get("input_text") or kwargs.get("inputs") or kwargs.get("text")
    if golds is None or inputs is None:
        raise ValueError("reward requires gold and input_text columns")
    return [extraction_reward(completion, gold, input_text) for completion, gold, input_text in zip(completions, golds, inputs)]
