import json

import pytest

from task_adapter import PHEE_SCHEMA, get_adapter, load_task_card


def test_loads_json_compatible_yaml_task_card(tmp_path):
    path = tmp_path / "task_card.yaml"
    path.write_text(json.dumps({"adapter": "phee"}))
    card = load_task_card(path)
    adapter = get_adapter(card)
    assert adapter.name == "phee"
    assert adapter.output_schema == PHEE_SCHEMA
    assert "Example sentence" in adapter.prompt("Example sentence")
    messages = adapter.prompt_messages("Example sentence")
    assert "verbatim" in messages[0]["content"]
    assert json.dumps(PHEE_SCHEMA, separators=(",", ":")) in messages[0]["content"]
    assert all(message["role"] != "assistant" for message in messages)


def test_phee_adapter_metric_is_reward():
    adapter = get_adapter({"adapter": "phee"})
    empty = {"events": []}
    assert adapter.metric(empty, empty, "No event is present.") == 1.0


def test_custom_adapter_requires_explicit_contract():
    with pytest.raises(ValueError, match="missing"):
        get_adapter({"adapter": "custom", "task": {"prompt_template": "{input}"}})


def test_custom_unknown_metric_is_rejected():
    card = {
        "adapter": "custom",
        "task": {
            "prompt_template": "{input}",
            "output_schema": {"type": "object"},
            "metric_name": "accuracy",
        },
    }
    with pytest.raises(ValueError, match="implemented"):
        get_adapter(card)
