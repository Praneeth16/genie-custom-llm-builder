import sys
import types

from evaluate import fmapi_predictions, local_predictions


def test_local_evaluation_uses_chat_and_lora_request(monkeypatch):
    captured = {}

    class Output:
        outputs = [types.SimpleNamespace(text='{"events": []}')]

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def chat(self, prompts, sampling_params, lora_request=None):
            captured["prompts"] = prompts
            captured["lora_request"] = lora_request
            return [Output()]

    class FakeLoRARequest:
        def __init__(self, name, identifier, path):
            self.path = path

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **kwargs: kwargs
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request)

    prompts = [[{"role": "system", "content": "rules"}, {"role": "user", "content": "input"}]]
    predictions, _, usage = local_predictions("base", "/adapter", prompts)
    assert predictions == ['{"events": []}']
    assert usage == {}
    assert captured["init"] == {"model": "base", "enable_lora": True, "max_lora_rank": 64}
    assert captured["prompts"] == prompts
    assert captured["lora_request"].path == "/adapter"


def test_fmapi_evaluation_uses_concurrency_retries_and_collects_usage(monkeypatch):
    attempts = {}

    class RetryableError(Exception):
        status_code = 429

    class FakeClient:
        def predict(self, endpoint, inputs):
            content = inputs["messages"][-1]["content"]
            attempts[content] = attempts.get(content, 0) + 1
            if content == "first" and attempts[content] == 1:
                raise RetryableError()
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }

    monkeypatch.setattr("evaluate.time.sleep", lambda _: None)
    prompts = [[{"role": "user", "content": value}] for value in ("first", "second")]
    predictions, _, usage = fmapi_predictions("endpoint", prompts, concurrency=2, client=FakeClient())
    assert predictions == ["first", "second"]
    assert attempts == {"first": 2, "second": 1}
    assert usage == {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
