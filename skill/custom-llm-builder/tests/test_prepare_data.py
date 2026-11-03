from prepare_data import convert, convert_with_diagnostics


def test_convert_keeps_prompt_and_completion_separate():
    row = convert({
        "sentence": "Aspirin caused nausea.",
        "events": [{"event_type": "Adverse_event", "trigger": "caused", "subject": [], "treatment": ["Aspirin"], "effect": ["nausea"]}],
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
        "subject": ["The patient"],
        "treatment": ["oral aspirin"],
        "effect": ["nausea"],
    }]}


def test_convert_represents_discontinuous_and_multiple_roles_as_lists():
    row, diagnostics = convert_with_diagnostics({
        "context": "Aspirin and ibuprofen caused nausea.",
        "annotations": [{"events": [{
            "event_type": "Adverse_event",
            "Trigger": {"text": [["caused"]]},
            "Treatment": {"text": [["Aspirin", "ibuprofen"], ["Aspirin"]]},
            "Effect": {"text": [["nausea"]]},
        }]}],
    })
    assert row["gold"]["events"][0]["treatment"] == ["Aspirin", "ibuprofen"]
    assert diagnostics == {
        "represented_missing_subject_as_empty_list": 1,
        "represented_multiple_treatment_entities_as_list": 1,
        "represented_discontinuous_treatment_pieces_as_list": 1,
        "represented_duplicate_treatment_span_once": 1,
    }


def test_convert_missing_effect_is_empty_list():
    row, diagnostics = convert_with_diagnostics({
        "context": "Aspirin treatment.",
        "annotations": [{"events": [{
            "event_type": "Potential_therapeutic_event",
            "Trigger": {"text": [["treatment"]]},
            "Treatment": {"text": [["Aspirin"]]},
        }]}],
    })
    assert row["gold"]["events"][0]["effect"] == []
    assert diagnostics["represented_missing_effect_as_empty_list"] == 1


def test_convert_drops_only_nonverbatim_span():
    row, diagnostics = convert_with_diagnostics({
        "context": "Aspirin caused nausea.",
        "annotations": [{"events": [{
            "event_type": "Adverse_event",
            "Trigger": {"text": [["caused"]]},
            "Treatment": {"text": [["Aspirin"]]},
            "Effect": {"text": [["headache"]]},
        }]}],
    })
    assert row["gold"] == {"events": []}
    assert diagnostics["dropped_nonverbatim_effect"] == 1
    assert diagnostics["dropped_events"] == 1
