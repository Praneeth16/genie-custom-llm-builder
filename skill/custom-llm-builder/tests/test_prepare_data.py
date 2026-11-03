from prepare_data import convert


def test_convert_keeps_prompt_and_completion_separate():
    row = convert({
        "sentence": "Aspirin caused nausea.",
        "events": [{"event_type": "Adverse_event", "trigger": "caused", "subject": None, "treatment": "Aspirin", "effect": "nausea"}],
    })
    assert row["prompt"][-1] == {"role": "user", "content": "Input:\nAspirin caused nausea."}
    assert all(message["role"] != "assistant" for message in row["prompt"])
    assert row["completion"][0]["role"] == "assistant"
