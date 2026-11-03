from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PHEE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event_type", "trigger", "subject", "treatment", "effect"],
                "properties": {
                    "event_type": {"enum": ["Adverse_event", "Potential_therapeutic_event"]},
                    "trigger": {"type": "string"},
                    "subject": {"type": ["string", "null"]},
                    "treatment": {"type": "string"},
                    "effect": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class TaskAdapter:
    name: str
    prompt_template: str
    output_schema: dict[str, Any]
    metric_name: str
    metric: Callable[..., float]

    def prompt(self, input_text: str) -> str:
        return self.prompt_template.format(input=input_text)


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
    from rewards import extraction_reward

    return extraction_reward(prediction, gold, input_text)


def get_adapter(task_card: dict[str, Any]) -> TaskAdapter:
    adapter_name = task_card.get("adapter", task_card.get("task", {}).get("adapter", "phee"))
    if adapter_name == "phee":
        return TaskAdapter(
            name="phee",
            prompt_template=(
                "Extract pharmacovigilance events from the sentence. Return only JSON matching "
                "the provided schema. Copy every extracted span verbatim.\nSentence: {input}"
            ),
            output_schema=PHEE_SCHEMA,
            metric_name="strict_event_f1",
            metric=_phee_metric,
        )
    custom = task_card.get("task", task_card)
    required = ("prompt_template", "output_schema", "metric_name")
    missing = [key for key in required if key not in custom]
    if missing:
        raise ValueError(f"custom task card is missing: {', '.join(missing)}")
    metric_name = custom["metric_name"]
    if metric_name != "strict_event_f1":
        raise ValueError("custom metrics must be implemented and registered before training")
    return TaskAdapter(
        name=str(adapter_name),
        prompt_template=custom["prompt_template"],
        output_schema=custom["output_schema"],
        metric_name=metric_name,
        metric=_phee_metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_card")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    adapter = get_adapter(load_task_card(args.task_card))
    print(adapter.prompt(args.input))


if __name__ == "__main__":
    main()
