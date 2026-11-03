from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any


EVENT_KEYS = {"event_type", "trigger", "subject", "treatment", "effect"}
EVENT_TYPES = {"Adverse_event", "Potential_therapeutic_event"}
SCORED_FIELDS = ("trigger", "treatment", "effect")
LIST_FIELDS = ("subject", "treatment", "effect")


def parse_document(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0].get("content")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def is_valid_document(document: Any, input_text: str) -> bool:
    if not isinstance(document, dict) or set(document) != {"events"} or not isinstance(document["events"], list):
        return False
    for event in document["events"]:
        if not isinstance(event, dict) or set(event) != EVENT_KEYS:
            return False
        if event["event_type"] not in EVENT_TYPES or not isinstance(event["trigger"], str):
            return False
        if any(not isinstance(event[field], list) for field in LIST_FIELDS):
            return False
        if any(any(not isinstance(span, str) for span in event[field]) for field in LIST_FIELDS):
            return False
        if any(len(event[field]) != len(set(event[field])) for field in LIST_FIELDS):
            return False
        spans = [event["trigger"]] + [span for field in LIST_FIELDS for span in event[field]]
        if any(span not in input_text for span in spans):
            return False
    return True


def validate_gold(gold: Any, input_text: str) -> dict[str, Any]:
    document = parse_document(gold)
    if not is_valid_document(document, input_text):
        raise ValueError("gold output is invalid or contains a non-verbatim span")
    return document


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold())


def token_f1(left: str, right: str) -> float:
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    remaining, overlap = list(right_tokens), 0
    for token in left_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    precision, recall = overlap / len(left_tokens), overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def role_items(event: dict[str, Any], field: str) -> list[str]:
    return [event[field]] if field == "trigger" else event[field]


def role_token_f1(left: list[str], right: list[str]) -> float:
    return token_f1(" ".join(left), " ".join(right))


def role_exact(left: list[str], right: list[str]) -> float:
    return float(set(left) == set(right))


def role_score(predicted: dict[str, Any], gold: dict[str, Any], field: str) -> float:
    predicted_items = role_items(predicted, field)
    gold_items = role_items(gold, field)
    return 0.9 * role_token_f1(predicted_items, gold_items) + 0.1 * role_exact(predicted_items, gold_items)


def event_score(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    if predicted["event_type"] != gold["event_type"]:
        return 0.0
    return sum(role_score(predicted, gold, field) for field in SCORED_FIELDS) / len(SCORED_FIELDS)


def maximum_matching(scores: list[list[float]]) -> tuple[float, list[tuple[int, int]]]:
    if not scores or not scores[0]:
        return 0.0, []
    if len(scores[0]) > 20:
        raise ValueError("too many gold events for exact one-to-one matching")

    @lru_cache(maxsize=None)
    def best(predicted_index: int, used_gold: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if predicted_index == len(scores):
            return 0.0, ()
        result = best(predicted_index + 1, used_gold)
        for gold_index, score in enumerate(scores[predicted_index]):
            if not used_gold & 1 << gold_index:
                tail_score, pairs = best(predicted_index + 1, used_gold | 1 << gold_index)
                candidate = (score + tail_score, ((predicted_index, gold_index),) + pairs)
                if candidate[0] > result[0]:
                    result = candidate
        return result

    total, pairs = best(0, 0)
    return total, list(pairs)


def extraction_metrics(prediction: Any, gold: Any, input_text: str) -> dict[str, float]:
    gold_document = validate_gold(gold, input_text)
    predicted_document = parse_document(prediction)
    zero = {
        "strict_event_f1": 0.0,
        "valid_format": 0.0,
        "empty_output": 0.0,
        "event_precision": 0.0,
        "event_recall": 0.0,
        "main_argument_exact_match": 0.0,
        "main_argument_token_f1": 0.0,
        "subject_exact_match": 0.0,
        "subject_token_f1": 0.0,
    }
    if not is_valid_document(predicted_document, input_text):
        return zero
    predicted, targets = predicted_document["events"], gold_document["events"]
    if not predicted and not targets:
        return {key: 1.0 for key in zero}
    scores = [[event_score(item, target) for target in targets] for item in predicted]
    matched, pairs = maximum_matching(scores)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(targets) if targets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    event_denominator = max(len(predicted), len(targets))
    main_denominator = event_denominator * len(SCORED_FIELDS)
    exact = sum(
        role_exact(role_items(predicted[i], field), role_items(targets[j], field))
        for i, j in pairs for field in SCORED_FIELDS
    ) / main_denominator if main_denominator else 0.0
    argument_f1 = sum(
        role_token_f1(role_items(predicted[i], field), role_items(targets[j], field))
        for i, j in pairs for field in SCORED_FIELDS
    ) / main_denominator if main_denominator else 0.0
    subject_exact = sum(
        role_exact(predicted[i]["subject"], targets[j]["subject"]) for i, j in pairs
    ) / event_denominator if event_denominator else 0.0
    subject_f1 = sum(
        role_token_f1(predicted[i]["subject"], targets[j]["subject"]) for i, j in pairs
    ) / event_denominator if event_denominator else 0.0
    return {
        "strict_event_f1": f1,
        "valid_format": 1.0,
        "empty_output": float(not predicted),
        "event_precision": precision,
        "event_recall": recall,
        "main_argument_exact_match": exact,
        "main_argument_token_f1": argument_f1,
        "subject_exact_match": subject_exact,
        "subject_token_f1": subject_f1,
    }
