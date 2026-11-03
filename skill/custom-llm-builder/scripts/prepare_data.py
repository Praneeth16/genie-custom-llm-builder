from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from metrics import validate_gold
from task_adapter import get_adapter


REPOSITORY = "https://github.com/ZhaoyueSun/phee-with-chatgpt.git"


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def find_records(root: Path, split: str) -> list[dict[str, Any]]:
    candidates = sorted(path for path in root.rglob("*.json*") if split in path.name.casefold())
    records: list[dict[str, Any]] = []
    for path in candidates:
        if path.suffix == ".jsonl":
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        else:
            value = json.loads(path.read_text())
            records.extend(value if isinstance(value, list) else value.get("data", []))
    return records


def first(record: dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def convert(record: dict[str, Any]) -> dict[str, Any]:
    text = first(record, ("sentence", "text", "context"))
    raw_events = first(record, ("events", "event", "annotations"), [])
    if isinstance(raw_events, dict):
        raw_events = [raw_events]
    events = []
    for event in raw_events:
        events.append({
            "event_type": first(event, ("event_type", "type")),
            "trigger": first(event, ("trigger",)),
            "subject": first(event, ("subject",), None),
            "treatment": first(event, ("treatment", "drug")),
            "effect": first(event, ("effect", "adverse_effect")),
        })
    answer = {"events": events}
    validate_gold(answer, text)
    adapter = get_adapter({"adapter": "phee"})
    return {
        "input_text": text,
        "gold": answer,
        "prompt": adapter.prompt_messages(text),
        "completion": [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}],
    }


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
    for split in ("train", "dev", "test"):
        rows = [convert(record) for record in find_records(source, split)]
        if not rows:
            raise RuntimeError(f"no {split} records found; inspect the upstream V2 layout")
        split_rows[split] = rows
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hashes = {split: {normalized_hash(row["input_text"]) for row in rows} for split, rows in split_rows.items()}
    overlaps = {f"{left}_{right}": len(hashes[left] & hashes[right]) for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    card = {"source": REPOSITORY, "version": "V2", "license": "MIT", "counts": {key: len(value) for key, value in split_rows.items()}, "normalized_exact_overlap": overlaps}
    (args.output / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
    if args.source is None:
        shutil.rmtree(source)


if __name__ == "__main__":
    main()
