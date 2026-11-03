from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from metrics import validate_gold
from task_adapter import get_adapter


REPOSITORY = "https://github.com/ZhaoyueSun/phee-with-chatgpt.git"


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def find_records(root: Path, split: str) -> list[dict[str, Any]]:
    source_path = root / "data" / "datasets" / "phee2" / "source" / f"{split}.json"
    if source_path.exists():
        return [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    candidates = sorted(path for path in root.rglob("*.json*") if split in path.name.casefold())
    records: list[dict[str, Any]] = []
    for path in candidates:
        if path.suffix == ".jsonl":
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        else:
            try:
                value = json.loads(path.read_text())
            except json.JSONDecodeError:
                records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
                continue
            records.extend(value if isinstance(value, list) else value.get("data", []))
    return records


def first(record: dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def source_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "annotations" in record and record["annotations"] and "events" in record["annotations"][0]:
        return [event for annotation in record["annotations"] for event in annotation.get("events", [])]
    raw_events = first(record, ("events", "event", "annotations"), [])
    return [raw_events] if isinstance(raw_events, dict) else raw_events


def source_span(field: Any, role: str, text: str, required: bool, reasons: Counter[str]) -> str | None:
    if field is None:
        if required:
            reasons[f"missing_{role}"] += 1
        return None
    if isinstance(field, str):
        value = field
    elif isinstance(field, dict):
        entities = field.get("text") or []
        if len(entities) != 1:
            reasons[f"multiple_{role}"] += 1
            return None
        pieces = entities[0]
        if len(pieces) != 1:
            reasons[f"discontinuous_{role}"] += 1
            return None
        value = pieces[0]
    else:
        reasons[f"invalid_{role}_shape"] += 1
        return None
    if value not in text:
        reasons[f"nonverbatim_{role}"] += 1
        return None
    return value


def convert_with_diagnostics(record: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    text = first(record, ("sentence", "text", "context"))
    reasons: Counter[str] = Counter()
    events = []
    for event in source_events(record):
        event_reasons: Counter[str] = Counter()
        nested = "Trigger" in event or "Treatment" in event or "Effect" in event
        if nested:
            trigger = source_span(event.get("Trigger"), "trigger", text, True, event_reasons)
            subject = source_span(event.get("Subject"), "subject", text, False, event_reasons)
            treatment = source_span(event.get("Treatment"), "treatment", text, True, event_reasons)
            effect = source_span(event.get("Effect"), "effect", text, True, event_reasons)
            event_type = event.get("event_type")
        else:
            trigger = source_span(first(event, ("trigger",), None), "trigger", text, True, event_reasons)
            subject = source_span(first(event, ("subject",), None), "subject", text, False, event_reasons)
            treatment = source_span(first(event, ("treatment", "drug"), None), "treatment", text, True, event_reasons)
            effect = source_span(first(event, ("effect", "adverse_effect"), None), "effect", text, True, event_reasons)
            event_type = first(event, ("event_type", "type"))
        event_type = {"adverse event": "Adverse_event", "potential therapeutic event": "Potential_therapeutic_event"}.get(event_type, event_type)
        if event_type not in {"Adverse_event", "Potential_therapeutic_event"}:
            event_reasons["invalid_event_type"] += 1
        required_invalid = any(
            key != "multiple_subject" and not key.endswith("_subject")
            for key in event_reasons
        )
        subject_invalid = any(key.endswith("_subject") for key in event_reasons)
        if required_invalid:
            reasons.update(event_reasons)
            reasons["dropped_events"] += 1
            continue
        if subject_invalid:
            subject = None
            reasons.update(event_reasons)
            reasons["subjects_mapped_to_null"] += 1
        events.append({
            "event_type": event_type,
            "trigger": trigger,
            "subject": subject,
            "treatment": treatment,
            "effect": effect,
        })
    answer = {"events": events}
    validate_gold(answer, text)
    adapter = get_adapter({"adapter": "phee"})
    return {
        "source_id": first(record, ("id", "text_id"), None),
        "input_text": text,
        "gold": answer,
        "prompt": adapter.prompt_messages(text),
        "completion": [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}],
    }, reasons


def convert(record: dict[str, Any]) -> dict[str, Any]:
    row, _ = convert_with_diagnostics(record)
    return row


def token_set(text: str) -> set[str]:
    return set(normalized_text(text).split())


def near_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]], threshold: float = 0.9) -> int:
    right_sets = [(normalized_text(row["input_text"]), token_set(row["input_text"])) for row in right]
    count = 0
    for row in left:
        left_normalized = normalized_text(row["input_text"])
        left_set = token_set(row["input_text"])
        for right_normalized, right_set in right_sets:
            if left_normalized == right_normalized:
                continue
            union = left_set | right_set
            if union and len(left_set & right_set) / len(union) >= threshold:
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source
    if source is None:
        source = args.output / "source"
        subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(source)], check=True)
    args.output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    retained_event_counts: dict[str, int] = {}
    invalid_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        records = find_records(source, split)
        if not records:
            raise RuntimeError(f"no {split} records found; inspect the upstream V2 layout")
        rows = []
        diagnostics: Counter[str] = Counter()
        source_counts[split] = len(records)
        event_counts[split] = sum(len(source_events(record)) for record in records)
        for record in records:
            row, row_diagnostics = convert_with_diagnostics(record)
            rows.append(row)
            diagnostics.update(row_diagnostics)
        retained_event_counts[split] = sum(len(row["gold"]["events"]) for row in rows)
        diagnostics["records_with_no_retained_events"] = sum(not row["gold"]["events"] for row in rows)
        invalid_by_split[split] = dict(sorted(diagnostics.items()))
        split_rows[split] = rows
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hashes = {split: {normalized_hash(row["input_text"]) for row in rows} for split, rows in split_rows.items()}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    exact = {f"{left}_{right}": len(hashes[left] & hashes[right]) for left, right in pairs}
    near = {f"{left}_{right}": near_overlap(split_rows[left], split_rows[right]) for left, right in pairs}
    totals = Counter()
    for diagnostics in invalid_by_split.values():
        totals.update(diagnostics)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    card = {
        "source": REPOSITORY,
        "version": "V2",
        "source_commit": commit,
        "license": "MIT",
        "source_record_counts": source_counts,
        "converted_record_counts": {key: len(value) for key, value in split_rows.items()},
        "source_event_counts": event_counts,
        "retained_event_counts": retained_event_counts,
        "invalid_gold_by_reason": invalid_by_split,
        "invalid_gold_totals": dict(sorted(totals.items())),
        "normalized_exact_overlap": exact,
        "near_duplicate_pairs": {"definition": "cross-split token-set Jaccard >= 0.90, excluding normalized exact matches", "counts": near},
    }
    (args.output / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
    with (args.output / "sample-20.jsonl").open("w", encoding="utf-8") as handle:
        for row in split_rows["dev"][:20]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.source is None:
        shutil.rmtree(source)


if __name__ == "__main__":
    main()
