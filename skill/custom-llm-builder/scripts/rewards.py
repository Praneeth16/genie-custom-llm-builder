from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any


EVENT_KEYS = {"event_type", "trigger", "subject", "treatment", "effect"}
EVENT_TYPES = {"Adverse_event", "Potential_therapeutic_event"}
SCORED_FIELDS = ("trigger", "treatment", "effect")


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _valid(document: Any, input_text: str) -> bool:
    if not isinstance(document, dict) or set(document) != {"events"}:
        return False
    events = document["events"]
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict) or set(event) != EVENT_KEYS:
            return False
        if event["event_type"] not in EVENT_TYPES or event["subject"] is not None and not isinstance(event["subject"], str):
            return False
        if any(not isinstance(event[field], str) for field in SCORED_FIELDS):
            return False
        spans = [event[field] for field in SCORED_FIELDS]
        if event["subject"] is not None:
            spans.append(event["subject"])
        if any(span not in input_text for span in spans):
            return False
    return True


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold())


def _token_f1(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    remaining = list(right_tokens)
    overlap = 0
    for token in left_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _event_score(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    if predicted["event_type"] != gold["event_type"]:
        return 0.0
    scores = []
    for field in SCORED_FIELDS:
        token_score = _token_f1(predicted[field], gold[field])
        scores.append(0.9 * token_score + 0.1 * float(predicted[field] == gold[field]))
    return sum(scores) / len(scores)


def _maximum_matching(scores: list[list[float]]) -> float:
    if not scores or not scores[0]:
        return 0.0
    if len(scores[0]) > 20:
        raise ValueError("too many gold events for exact one-to-one matching")

    @lru_cache(maxsize=None)
    def best(predicted_index: int, used_gold: int) -> float:
        if predicted_index == len(scores):
            return 0.0
        result = best(predicted_index + 1, used_gold)
        for gold_index, score in enumerate(scores[predicted_index]):
            if not used_gold & 1 << gold_index:
                result = max(result, score + best(predicted_index + 1, used_gold | 1 << gold_index))
        return result

    return best(0, 0)


def extraction_reward(prediction: Any, gold: Any, input_text: str) -> float:
    predicted_document, gold_document = _parse(prediction), _parse(gold)
    if not _valid(predicted_document, input_text) or not _valid(gold_document, input_text):
        return 0.0
    predicted_events, gold_events = predicted_document["events"], gold_document["events"]
    if not predicted_events and not gold_events:
        return 1.0
    if not predicted_events or not gold_events:
        return 0.0
    scores = [[_event_score(predicted, target) for target in gold_events] for predicted in predicted_events]
    matched = _maximum_matching(scores)
    precision, recall = matched / len(predicted_events), matched / len(gold_events)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def gated_extraction_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    golds = kwargs.get("gold") or kwargs.get("answer") or kwargs.get("targets")
    inputs = kwargs.get("input_text") or kwargs.get("inputs") or kwargs.get("text")
    if golds is None or inputs is None:
        raise ValueError("reward requires gold and input_text columns")
    return [extraction_reward(completion, gold, input_text) for completion, gold, input_text in zip(completions, golds, inputs)]
