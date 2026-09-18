"""LLM Provider router and factory."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from app.ai.chained import ChainedFallbackProvider
from app.ai.gemini import GeminiProvider
from app.ai.groq import GroqProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.provider import LLMProvider, LLMResponse
from app.core.config import get_settings
from app.core.exceptions import ConfigurationException
from app.core.logging import get_logger

logger = get_logger("ai.router")
T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """Mock LLM Provider for unit testing, offline development, and deterministic assertions."""

    def __init__(self, canned_response: str | None = None, canned_data: dict[str, Any] | None = None):
        self._canned_response = canned_response or "haan bol kya scene"
        self.canned_data = canned_data

    @property
    def canned_response(self) -> str:
        return self._canned_response

    @canned_response.setter
    def canned_response(self, val: str) -> None:
        self._canned_response = val
        if self.canned_data:
            self.canned_data["reply_text"] = val

    def set_response(self, text: str, intent: str = "REPLY") -> None:
        self._canned_response = text
        if self.canned_data:
            self.canned_data["reply_text"] = text
            self.canned_data["intent"] = intent

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        structured = dict(self.canned_data) if self.canned_data else {
            "intent": "REPLY",
            "reply_text": self._canned_response,
            "internal_reasoning": "Mocked deadpan reaction.",
            "humor_style_applied": "dry_sarcasm",
            "callback_referenced": None,
        }
        if "reply_text" in structured and structured["reply_text"] != self._canned_response:
            structured["reply_text"] = self._canned_response

        content_str = json.dumps(structured) if response_schema else self._canned_response
        return LLMResponse(
            content=content_str,
            structured_data=structured if response_schema else None,
            prompt_tokens=42,
            output_tokens=18,
            latency_ms=12.5,
            model="mock-v1",
            provider="mock",
        )

    async def health_check(self) -> bool:
        return True


_provider_instance: LLMProvider | None = None


def get_llm_provider(force_mock: bool = False) -> LLMProvider:
    """Resolve and cache the configured LLM provider."""
    global _provider_instance
    if _provider_instance is not None and not force_mock:
        return _provider_instance

    settings = get_settings()
    provider_name = "mock" if force_mock else settings.LLM_PROVIDER.lower()

    if provider_name in ("chained", "auto", "fallback"):
        _provider_instance = ChainedFallbackProvider()
    elif provider_name == "gemini":
        _provider_instance = GeminiProvider()
    elif provider_name == "groq":
        _provider_instance = GroqProvider()
    elif provider_name == "openrouter":
        _provider_instance = OpenRouterProvider()
    elif provider_name == "mock":
        _provider_instance = MockLLMProvider()
    else:
        raise ConfigurationException(f"Unsupported LLM provider: '{provider_name}'. Supported: 'chained', 'gemini', 'groq', 'openrouter', 'mock'.")

    logger.info("ai.provider_resolved", provider=provider_name)
    return _provider_instance

