from __future__ import annotations

import json
from typing import Any

import mlflow


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class ADEExtractor(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from task_adapter import get_adapter
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.adapter = get_adapter({"adapter": "phee"})
        self.engine = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64)
        self.sampling = SamplingParams(temperature=0, max_tokens=512)
        self.lora_request = LoRARequest("ade-sft-full-v2", 1, context.artifacts["adapter"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any, params: dict | None = None) -> list[str]:
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            values = model_input.get("sentence", model_input.get("input_text", []))
            records = [{"sentence": value} for value in values]
        else:
            records = list(model_input)
        sentences = [record.get("sentence", record.get("input_text")) for record in records]
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise ValueError("model input requires a string sentence or input_text column")
        prompts = [self.adapter.prompt_messages(sentence) for sentence in sentences]
        outputs = self.engine.chat(prompts, self.sampling, lora_request=self.lora_request)
        return [output.outputs[0].text for output in outputs]


def validate_json_string(value: str) -> bool:
    parsed = json.loads(value)
    return isinstance(parsed, dict) and isinstance(parsed.get("events"), list)
