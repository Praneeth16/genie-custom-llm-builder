import json

import pytest

from rewards import extraction_reward, gated_extraction_reward


TEXT = "The patient developed severe nausea after aspirin treatment."
EVENT = {
    "event_type": "Adverse_event",
    "trigger": "developed",
    "subject": "The patient",
    "treatment": "aspirin",
    "effect": "severe nausea",
}
GOLD = {"events": [EVENT]}


def test_invalid_json_is_zero():
    assert extraction_reward("not json", GOLD, TEXT) == 0.0


def test_string_not_in_input_is_zero():
    prediction = {"events": [{**EVENT, "effect": "headache"}]}
    assert extraction_reward(prediction, GOLD, TEXT) == 0.0


def test_both_empty_is_one():
    assert extraction_reward({"events": []}, {"events": []}, TEXT) == 1.0


def test_empty_prediction_against_gold_is_zero():
    assert extraction_reward({"events": []}, GOLD, TEXT) == 0.0


def test_copying_whole_sentence_into_every_span_scores_poorly():
    copied = {"events": [{**EVENT, "trigger": TEXT, "subject": TEXT, "treatment": TEXT, "effect": TEXT}]}
    score = extraction_reward(copied, GOLD, TEXT)
    assert 0.0 < score < 0.35


def test_duplicate_events_reduce_precision():
    prediction = {"events": [EVENT, EVENT]}
    assert extraction_reward(prediction, GOLD, TEXT) == pytest.approx(2 / 3)


def test_partial_match_gets_partial_credit():
    prediction = {"events": [{**EVENT, "effect": "nausea"}]}
    score = extraction_reward(prediction, GOLD, TEXT)
    assert 0.0 < score < 1.0


def test_off_schema_output_is_zero():
    prediction = {"events": [{**EVENT, "confidence": 1.0}]}
    assert extraction_reward(prediction, GOLD, TEXT) == 0.0


def test_trl_reward_signature_batches_rows():
    scores = gated_extraction_reward([json.dumps(GOLD), '{"events": []}'], gold=[GOLD, GOLD], input_text=[TEXT, TEXT])
    assert scores == [1.0, 0.0]


def test_trl_conversational_completion_is_parsed():
    completion = [{"role": "assistant", "content": json.dumps(GOLD)}]
    assert gated_extraction_reward([completion], gold=[GOLD], input_text=[TEXT]) == [1.0]


def test_invalid_gold_fails_loudly():
    with pytest.raises(ValueError, match="gold output"):
        extraction_reward(GOLD, {"events": [{**EVENT, "effect": "fabricated"}]}, TEXT)
