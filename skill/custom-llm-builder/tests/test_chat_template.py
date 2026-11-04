import sys
import types

import pytest

from chat_template import use_reference_chat_template


class FakeTokenizer:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False):
        return self.chat_template.format(user=messages[0]["content"], assistant=messages[1]["content"])


def install_reference(monkeypatch, template):
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda model_id: FakeTokenizer(template))
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_reference_template_replaces_mirrored_template(monkeypatch):
    install_reference(monkeypatch, "U:{user} A:{assistant}")
    tokenizer = FakeTokenizer("U:{user} A:<think></think>{assistant}")
    use_reference_chat_template(tokenizer, "Qwen/Qwen3-4B-Instruct-2507")
    assert "<think>" not in tokenizer.apply_chat_template([{"content": "x"}, {"content": "y"}])


def test_mismatch_after_override_fails(monkeypatch):
    install_reference(monkeypatch, "U:{user} A:{assistant}")

    class Stubborn(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False):
            return "<think></think>"

    with pytest.raises(RuntimeError):
        use_reference_chat_template(Stubborn("ignored"), "Qwen/Qwen3-4B-Instruct-2507")
