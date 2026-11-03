from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import is_valid_document, parse_document
from task_adapter import get_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    adapter = get_adapter({"adapter": "phee"})
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    written, rejected = 0, 0
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = adapter.prompt_messages(row["input_text"])
            response = client.predict(endpoint=args.endpoint, inputs={"messages": prompt, "max_tokens": 512})
            content = response["choices"][0]["message"]["content"]
            if not is_valid_document(parse_document(content), row["input_text"]):
                rejected += 1
                continue
            handle.write(json.dumps({"input_text": row["input_text"], "prompt": prompt, "completion": [{"role": "assistant", "content": content}]}, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "rejected": rejected}))


if __name__ == "__main__":
    main()
