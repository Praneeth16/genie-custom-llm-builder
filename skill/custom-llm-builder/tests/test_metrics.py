import json
from metrics import extraction_metrics


TEXT = "The patient developed severe nausea after aspirin treatment."
EVENT = {"event_type": "Adverse_event", "trigger": "developed", "subject": ["The patient"], "treatment": ["aspirin"], "effect": ["severe nausea"]}


def test_metrics_report_perfect_prediction():
    metrics = extraction_metrics({"events": [EVENT]}, {"events": [EVENT]}, TEXT)
    assert metrics["strict_event_f1"] == 1.0
    assert metrics["event_precision"] == 1.0
    assert metrics["event_recall"] == 1.0
    assert metrics["main_argument_exact_match"] == 1.0
    assert metrics["main_argument_token_f1"] == 1.0
    assert metrics["subject_exact_match"] == 1.0
    assert metrics["subject_token_f1"] == 1.0
    assert metrics["empty_output"] == 0.0


def test_metrics_report_valid_empty_prediction():
    metrics = extraction_metrics({"events": []}, {"events": [EVENT]}, TEXT)
    assert metrics["valid_format"] == 1.0
    assert metrics["empty_output"] == 1.0
    assert metrics["strict_event_f1"] == 0.0


def test_main_argument_exact_match_uses_span_sets():
    reordered = {**EVENT, "treatment": ["treatment", "aspirin"]}
    gold = {**EVENT, "treatment": ["aspirin", "treatment"]}
    metrics = extraction_metrics({"events": [reordered]}, {"events": [gold]}, TEXT)
    assert metrics["main_argument_exact_match"] == 1.0


def test_leading_reasoning_block_is_ignored_but_other_text_is_not():
    document = json.dumps({"events": [EVENT]})
    assert extraction_metrics(f"<think>\n\n</think>\n\n{document}", {"events": [EVENT]}, TEXT)["strict_event_f1"] == 1.0
    assert extraction_metrics(f"Here is the JSON: {document}", {"events": [EVENT]}, TEXT)["valid_format"] == 0.0


def test_provider_content_blocks_use_only_text_blocks():
    document = json.dumps({"events": [EVENT]})
    blocks = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": ""}]}, {"type": "text", "text": document}]
    assert extraction_metrics(blocks, {"events": [EVENT]}, TEXT)["strict_event_f1"] == 1.0
