"""Configurable language-model boundary for research reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_LLM_CONFIG_PATH = Path("config/llm.json")


class LLMConfigurationError(ValueError):
    """Raised when an LLM runtime configuration is incomplete or invalid."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM request fails or returns unusable content."""


@dataclass(frozen=True)
class LLMConfig:
    """Runtime settings for one model provider without storing credentials."""

    provider: str
    model: str
    api_key_env: str
    base_url: str | None = None
    temperature: float = 0.1
    max_repair_attempts: int = 1

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LLM_CONFIG_PATH) -> "LLMConfig":
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise LLMConfigurationError(f"Cannot read LLM config: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(f"Invalid JSON in LLM config: {config_path}") from exc
        if not isinstance(data, Mapping):
            raise LLMConfigurationError("LLM config must contain a JSON object")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LLMConfig":
        allowed = {
            "provider", "model", "api_key_env", "base_url", "temperature",
            "max_repair_attempts",
        }
        unknown = set(data) - allowed
        if unknown:
            raise LLMConfigurationError(
                f"Unknown LLM config fields: {', '.join(sorted(unknown))}"
            )
        provider = _required_text(data, "provider")
        model = _required_text(data, "model")
        api_key_env = _required_text(data, "api_key_env")
        base_url = data.get("base_url")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
            raise LLMConfigurationError("'base_url' must be a non-empty string or null")
        temperature = data.get("temperature", 0.1)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise LLMConfigurationError("'temperature' must be a number")
        max_repair_attempts = data.get("max_repair_attempts", 1)
        if (
            not isinstance(max_repair_attempts, int)
            or isinstance(max_repair_attempts, bool)
            or max_repair_attempts < 0
        ):
            raise LLMConfigurationError("'max_repair_attempts' must be a non-negative integer")
        return cls(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url.strip() if isinstance(base_url, str) else None,
            temperature=float(temperature),
            max_repair_attempts=max_repair_attempts,
        )

    def api_key(self, environ: Mapping[str, str] | None = None) -> str:
        environment = os.environ if environ is None else environ
        value = environment.get(self.api_key_env)
        if not value:
            raise LLMConfigurationError(
                f"LLM API key environment variable is not set: {self.api_key_env}"
            )
        return value


class LLMProvider(Protocol):
    """Small provider contract needed by the planner."""

    def generate_json(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        """Return a parsed JSON object for the supplied conversation."""


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral function call requested by an assistant turn."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMAssistantTurn:
    """Provider-neutral assistant response used by the executor loop."""

    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    finish_reason: str | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ToolCallingLLMProvider(Protocol):
    """Provider contract required by the single-agent executor."""

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMAssistantTurn:
        """Return assistant text and zero or more requested tool calls."""


class OpenAICompatibleProvider:
    """Chat Completions adapter for OpenAI-compatible model endpoints."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        if config.provider != "openai_compatible":
            raise LLMConfigurationError(
                f"Unsupported provider for this adapter: {config.provider}"
            )
        self.config = config
        self._client = client

    def generate_json(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        client = self._client or self._create_client()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                temperature=self.config.temperature,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM response did not contain text")
        return _parse_json_object(content)

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMAssistantTurn:
        client = self._client or self._create_client()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                tools=list(tools),
                tool_choice="auto",
                temperature=self.config.temperature,
            )
            choice = response.choices[0]
            message = choice.message
            calls = tuple(
                LLMToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in (message.tool_calls or ())
            )
            return LLMAssistantTurn(
                content=message.content,
                tool_calls=calls,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except Exception as exc:
            raise LLMProviderError(f"LLM tool request failed: {exc}") from exc

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'openai' package is required for the OpenAI-compatible provider"
            ) from exc
        options: dict[str, str] = {"api_key": self.config.api_key()}
        if self.config.base_url is not None:
            options["base_url"] = self.config.base_url
        return OpenAI(**options)


def _parse_json_object(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("LLM response is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise LLMProviderError("LLM JSON response must be an object")
    return value


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigurationError(f"'{key}' must be a non-empty string")
    return value.strip()
