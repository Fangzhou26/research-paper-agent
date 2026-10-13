import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent import (
    LLMAssistantTurn,
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    OpenAICompatibleProvider,
)


def test_llm_config_loads_without_reading_credential(tmp_path: Path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(json.dumps({
        "provider": "openai_compatible",
        "model": "test-model",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "TEST_LLM_KEY",
    }), encoding="utf-8")

    config = LLMConfig.load(path)

    assert config.model == "test-model"
    assert config.api_key({"TEST_LLM_KEY": "secret"}) == "secret"
    with pytest.raises(LLMConfigurationError, match="TEST_LLM_KEY"):
        config.api_key({})


def test_provider_parses_json_fence_and_passes_runtime_settings() -> None:
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='```json\n{"answer": 42}\n```')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    config = LLMConfig("openai_compatible", "test-model", "TEST_KEY", temperature=0.2)
    provider = OpenAICompatibleProvider(config, client=client)

    assert provider.generate_json([{"role": "user", "content": "question"}]) == {"answer": 42}
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0.2


def test_provider_rejects_non_json_response() -> None:
    message = SimpleNamespace(content="not json")
    completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    ))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleProvider(
        LLMConfig("openai_compatible", "test-model", "TEST_KEY"), client=client
    )

    with pytest.raises(LLMProviderError, match="not valid JSON"):
        provider.generate_json([])


def test_provider_normalizes_tool_calls() -> None:
    function = SimpleNamespace(name="list_papers", arguments="{}")
    tool_call = SimpleNamespace(id="call-1", function=function)
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[choice]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleProvider(
        LLMConfig("openai_compatible", "test-model", "TEST_KEY"), client=client
    )

    turn = provider.complete_with_tools([], [])

    assert isinstance(turn, LLMAssistantTurn)
    assert turn.tool_calls[0].name == "list_papers"
    assert turn.to_message()["tool_calls"][0]["id"] == "call-1"
