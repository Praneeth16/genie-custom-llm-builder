"""Create (or sync) the Vector Search index over the frame embeddings table, then query it.

python scripts/vector_index.py --endpoint <existing endpoint> create
python scripts/vector_index.py query --frame 1234
"""

from __future__ import annotations

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    VectorIndexType,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "query"))
    parser.add_argument("--endpoint")
    parser.add_argument("--frame", type=int, default=1234, help="frame_id to find neighbours of")
    args = parser.parse_args()

    schema = f"{os.environ['JEPA_CATALOG']}.{os.environ['JEPA_SCHEMA']}"
    table, index = f"{schema}.frame_embeddings", f"{schema}.frame_embeddings_index"
    client = WorkspaceClient(profile=os.environ["DATABRICKS_PROFILE"])

    if args.action == "create":
        client.vector_search_indexes.create_index(
            name=index,
            endpoint_name=args.endpoint,
            primary_key="frame_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=table,
                pipeline_type=PipelineType.TRIGGERED,
                embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=1024)],
            ),
        )
        print(f"created {index}; it syncs in the background")
        return

    columns = ["frame_id", "episode", "frame_in_episode", "agent_x", "agent_y"]
    row = client.vector_search_indexes.query_index(
        index_name=index, columns=[*columns, "embedding"], filters_json=f'{{"frame_id": {args.frame}}}', num_results=1
    ).result.data_array[0]
    print("query frame:", dict(zip(columns, row[:5], strict=False)))
    hits = client.vector_search_indexes.query_index(
        index_name=index, columns=columns, query_vector=[float(v) for v in row[5]], num_results=6
    )
    for hit in [h for h in hits.result.data_array if int(h[0]) != args.frame][:5]:
        print(dict(zip([*columns, "score"], hit, strict=False)))


if __name__ == "__main__":
    main()
