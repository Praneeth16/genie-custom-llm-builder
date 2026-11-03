import sys
import types

from evaluate import local_predictions


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
    predictions, _ = local_predictions("base", "/adapter", prompts)
    assert predictions == ['{"events": []}']
    assert captured["init"] == {"model": "base", "enable_lora": True, "max_lora_rank": 64}
    assert captured["prompts"] == prompts
    assert captured["lora_request"].path == "/adapter"
