from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import mlflow
import pandas as pd


DEFAULT_MODEL = "models:/serverless_lakebase_praneeth_catalog.custom_llm.ade_extractor@champion"
DEFAULT_OUTPUT = "serverless_lakebase_praneeth_catalog.custom_llm.ade_events"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def statement(client, warehouse_id: str, sql: str):
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="50s",
    )
    state = response.status.state.value
    while state in {"PENDING", "RUNNING"}:
        time.sleep(2)
        response = client.statement_execution.get_statement(response.statement_id)
        state = response.status.state.value
    if state != "SUCCEEDED":
        raise RuntimeError(f"SQL statement failed: {response.status}")
    return response


def read_delta(client, warehouse_id: str, table: str, limit: int | None) -> list[dict]:
    suffix = f" LIMIT {limit}" if limit else ""
    response = statement(client, warehouse_id, f"SELECT sentence FROM {table}{suffix}")
    values = response.result.data_array or []
    return [{"sentence": row[0]} for row in values]


def read_jsonl(path: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = rows[:limit] if limit else rows
    return [{"sentence": row.get("sentence", row.get("input_text"))} for row in rows]


def write_delta(client, warehouse_id: str, table: str, rows: list[dict]) -> None:
    statement(client, warehouse_id, f"""
        CREATE TABLE IF NOT EXISTS {table} (
          score_id STRING, sentence STRING, events_json STRING, parse_valid BOOLEAN,
          model_uri STRING, scored_at TIMESTAMP
        ) USING DELTA
    """)
    values = ",\n".join(
        f"({sql_literal(row['score_id'])}, {sql_literal(row['sentence'])}, {sql_literal(row['events_json'])}, "
        f"{str(row['parse_valid']).upper()}, {sql_literal(row['model_uri'])}, current_timestamp())"
        for row in rows
    )
    statement(client, warehouse_id, f"""
        MERGE INTO {table} target
        USING (SELECT * FROM VALUES {values}
          AS source(score_id, sentence, events_json, parse_valid, model_uri, scored_at)) source
        ON target.score_id = source.score_id
        WHEN NOT MATCHED THEN INSERT *
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl")
    source.add_argument("--input-table")
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL)
    parser.add_argument("--warehouse-id", default=os.environ.get("WAREHOUSE_ID"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--report-output")
    args = parser.parse_args()
    if not args.warehouse_id:
        parser.error("--warehouse-id or WAREHOUSE_ID is required")

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    inputs = read_jsonl(args.input_jsonl, args.limit) if args.input_jsonl else read_delta(client, args.warehouse_id, args.input_table, args.limit)
    model = mlflow.pyfunc.load_model(args.model_uri)
    predictions = model.predict(pd.DataFrame(inputs))
    output = []
    for index, (item, prediction) in enumerate(zip(inputs, predictions)):
        try:
            parsed = json.loads(prediction)
            valid = isinstance(parsed, dict) and isinstance(parsed.get("events"), list)
        except (TypeError, json.JSONDecodeError):
            valid = False
        output.append({
            "score_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.run_key}:{index}:{item['sentence']}")),
            "sentence": item["sentence"],
            "events_json": prediction,
            "parse_valid": valid,
            "model_uri": args.model_uri,
        })
    write_delta(client, args.warehouse_id, args.output_table, output)
    report = {
        "rows_written": len(output),
        "parse_valid_rows": sum(row["parse_valid"] for row in output),
        "parse_rate": sum(row["parse_valid"] for row in output) / len(output) if output else 0,
        "output_table": args.output_table,
        "model_uri": args.model_uri,
    }
    if args.report_output:
        Path(args.report_output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if report["parse_rate"] != 1:
        raise RuntimeError(f"not every batch output parsed: {report}")


if __name__ == "__main__":
    main()
