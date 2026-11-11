"""Unity Catalog helpers that work from AI Runtime, where there is no Spark session."""

from __future__ import annotations


def table_from_parquet(table: str, volume_path: str, warehouse_id: str) -> None:
    """Replace `table` with the contents of a parquet file in a UC Volume, via a SQL warehouse."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    statements = (
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_files('{volume_path}', format => 'parquet')",
        # Vector Search's Delta Sync index needs the change data feed.
        f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    )
    client = WorkspaceClient()
    for statement in statements:
        result = client.statement_execution.execute_statement(statement, warehouse_id, wait_timeout="50s")
        while result.status.state in (StatementState.PENDING, StatementState.RUNNING):
            result = client.statement_execution.get_statement(result.statement_id)
        if result.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"{statement[:60]}... failed: {result.status.error}")
