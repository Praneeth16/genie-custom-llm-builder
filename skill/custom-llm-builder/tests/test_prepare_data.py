from prepare_data import convert, convert_with_diagnostics


def test_convert_keeps_prompt_and_completion_separate():
    row = convert({
        "sentence": "Aspirin caused nausea.",
        "events": [{"event_type": "Adverse_event", "trigger": "caused", "subject": None, "treatment": "Aspirin", "effect": "nausea"}],
    })
    assert row["prompt"][-1] == {"role": "user", "content": "Input:\nAspirin caused nausea."}
    assert all(message["role"] != "assistant" for message in row["prompt"])
    assert row["completion"][0]["role"] == "assistant"


def test_convert_real_phee_v2_record_shape():
    row, diagnostics = convert_with_diagnostics({
        "id": "8908396_3",
        "context": "The patient developed nausea after oral aspirin.",
        "annotations": [{"events": [{
            "event_id": "E1",
            "event_type": "Adverse_event",
            "Trigger": {"text": [["developed"]], "start": [[12]]},
            "Subject": {"text": [["The patient"]], "start": [[0]]},
            "Treatment": {"text": [["oral aspirin"]], "start": [[35]]},
            "Effect": {"text": [["nausea"]], "start": [[22]]},
        }]}],
    })
    assert diagnostics == {}
    assert row["source_id"] == "8908396_3"
    assert row["gold"] == {"events": [{
        "event_type": "Adverse_event",
        "trigger": "developed",
        "subject": "The patient",
        "treatment": "oral aspirin",
        "effect": "nausea",
    }]}


def test_convert_reports_and_drops_incompatible_required_span():
    row, diagnostics = convert_with_diagnostics({
        "context": "Aspirin and ibuprofen caused nausea.",
        "annotations": [{"events": [{
            "event_type": "Adverse_event",
            "Trigger": {"text": [["caused"]]},
            "Treatment": {"text": [["Aspirin", "ibuprofen"]]},
            "Effect": {"text": [["nausea"]]},
        }]}],
    })
    assert row["gold"] == {"events": []}
    assert diagnostics == {"discontinuous_treatment": 1, "dropped_events": 1}
