"""Create or update a Model Serving endpoint for a registered champion, or send it a test request.

python scripts/serve.py create sensor
python scripts/serve.py create encoder
python scripts/serve.py query sensor
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

# The sensor model is tiny and runs on CPU. The encoder is a 300M-parameter ViT and wants a GPU.
MODELS = {
    "sensor": ("sensor_jepa", "jepa-wm-sensor", ServingModelWorkloadType.CPU),
    "encoder": ("vjepa2_encoder", "jepa-wm-encoder", ServingModelWorkloadType.GPU_SMALL),
}


def example(which: str) -> list[dict]:
    if which == "sensor":
        from jepa_wm.data.synthetic_sensor import generate, windows

        x, _ = windows(*generate(2000, anomaly_fraction=0.05, seed=7), stride=128)
        return [{"window": w.tolist()} for w in x[:4]]
    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((96, 96, 3), 255, dtype=np.uint8)).save(buffer, format="PNG")
    return [{"image": base64.b64encode(buffer.getvalue()).decode()}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "query"))
    parser.add_argument("model", choices=tuple(MODELS))
    args = parser.parse_args()

    model, endpoint, workload = MODELS[args.model]
    name = f"{os.environ['JEPA_CATALOG']}.{os.environ['JEPA_SCHEMA']}.{model}"
    client = WorkspaceClient(profile=os.environ["DATABRICKS_PROFILE"])

    if args.action == "create":
        version = client.model_versions.get_by_alias(name, "champion").version
        entities = [
            ServedEntityInput(
                entity_name=name,
                entity_version=str(version),
                workload_type=workload,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
        if endpoint in {e.name for e in client.serving_endpoints.list()}:
            client.serving_endpoints.update_config(name=endpoint, served_entities=entities)
        else:
            client.serving_endpoints.create(
                name=endpoint, config=EndpointCoreConfigInput(name=endpoint, served_entities=entities)
            )
        print(f"{endpoint}: deploying {name} v{version} on {workload.value}; this takes 10 to 30 minutes")
        return

    started = time.perf_counter()
    response = client.serving_endpoints.query(endpoint, dataframe_records=example(args.model))
    seconds = time.perf_counter() - started
    for row in response.predictions:
        print({k: (f"<{len(v)} floats>" if isinstance(v, list) else round(v, 3)) for k, v in row.items()})
    print(f"{seconds:.2f} s for {len(response.predictions)} rows")


if __name__ == "__main__":
    main()
