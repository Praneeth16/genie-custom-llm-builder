from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PHEE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["events"],
    "properties": {"events": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["event_type", "trigger", "subject", "treatment", "effect"],
        "properties": {
            "event_type": {"enum": ["Adverse_event", "Potential_therapeutic_event"]},
            "trigger": {"type": "string"},
            "subject": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "treatment": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "effect": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        },
    }}},
}


@dataclass(frozen=True)
class TaskAdapter:
    name: str
    instruction: str
    output_schema: dict[str, Any]
    metric_name: str
    metric: Callable[..., float]

    def prompt_messages(self, input_text: str, examples: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.instruction}]
        for example in examples or []:
            messages.extend(self.example_messages(example["input_text"], example["gold"]))
        messages.append({"role": "user", "content": f"Input:\n{input_text}"})
        return messages

    def example_messages(self, input_text: str, gold: Any) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": f"Input:\n{input_text}"},
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
        ]

    def prompt(self, input_text: str) -> str:
        return self.prompt_messages(input_text)[-1]["content"]


def load_task_card(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install PyYAML to load non-JSON task cards") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("task card must be a mapping")
    return data


def _phee_metric(prediction: Any, gold: Any, input_text: str) -> float:
    from metrics import extraction_metrics
    return extraction_metrics(prediction, gold, input_text)["strict_event_f1"]


def get_adapter(task_card: dict[str, Any]) -> TaskAdapter:
    adapter_name = task_card.get("adapter", task_card.get("task", {}).get("adapter", "phee"))
    if adapter_name == "phee":
        schema = json.dumps(PHEE_SCHEMA, separators=(",", ":"))
        return TaskAdapter(
            "phee",
            f"Extract pharmacovigilance events. Return only JSON matching this schema: {schema}. Copy every extracted string verbatim from the input. Use lists for subject, treatment, and effect, for example: {{\"subject\":[],\"treatment\":[\"aspirin\"],\"effect\":[\"nausea\"]}}.",
            PHEE_SCHEMA, "strict_event_f1", _phee_metric,
        )
    custom = task_card.get("task", task_card)
    required = ("prompt_template", "output_schema", "metric_name")
    missing = [key for key in required if key not in custom]
    if missing:
        raise ValueError(f"custom task card is missing: {', '.join(missing)}")
    if custom["metric_name"] != "strict_event_f1":
        raise ValueError("custom metrics must be implemented and registered before training")
    return TaskAdapter(str(adapter_name), custom["prompt_template"], custom["output_schema"], custom["metric_name"], _phee_metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_card")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(get_adapter(load_task_card(args.task_card)).prompt_messages(args.input)))


if __name__ == "__main__":
    main()
