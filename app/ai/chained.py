"""Chained Multi-Tier LLM Provider with automatic failover across different backends."""

from __future__ import annotations

import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from app.ai.gemini import GeminiProvider
from app.ai.groq import GroqProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.provider import LLMProvider, LLMResponse
from app.core.config import get_settings
from app.core.exceptions import LLMException
from app.core.logging import get_logger

logger = get_logger("ai.chained")
T = TypeVar("T", bound=BaseModel)


class ChainedFallbackProvider(LLMProvider):
    """Multi-tier LLM Provider that cascades across distinct AI backends.

    Fallback Order:
      1. OpenRouter Qwen (primary) -> qwen/qwen3.8-27b:free (auto-fallback to paid slug qwen/qwen3.8-27b)
      2. Gemini (secondary) -> gemini-3.6-flash (fallback to gemini-2.5-flash)
      3. OpenRouter Llama (tertiary) -> meta-llama/llama-3.3-70b-instruct:free (auto-fallback to paid slug)
    """

    def __init__(
        self,
        providers: Sequence[tuple[str, LLMProvider]] | None = None,
    ):
        settings = get_settings()
        if providers is not None:
            self.providers = list(providers)
        else:
            # Primary: OpenRouter model configured in settings (defaulting to NVIDIA Nemotron 3 Super)
            primary_model = settings.LLM_MODEL or "nvidia/nemotron-3-super-120b-a12b:free"
            primary_fallback = settings.LLM_FALLBACK_MODEL or "qwen/qwen3.8-27b"

            # Secondary: Gemini Flash
            gemini_model = "gemini-3.6-flash"
            gemini_fallback = "gemini-2.5-flash"
            if settings.LLM_PROVIDER.lower() == "gemini" and settings.LLM_MODEL.startswith("gemini"):
                gemini_model = settings.LLM_MODEL

            self.providers = [
                (
                    "openrouter-primary",
                    OpenRouterProvider(
                        model=primary_model,
                        fallback_model=primary_fallback,
                    ),
                ),
                (
                    "gemini",
                    GeminiProvider(
                        model=gemini_model,
                        fallback_model=gemini_fallback,
                    ),
                ),
                (
                    "openrouter-llama",
                    OpenRouterProvider(
                        model="meta-llama/llama-3.3-70b-instruct:free",
                        fallback_model="meta-llama/llama-3.3-70b-instruct",
                    ),
                ),
            ]

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for i, (name, provider) in enumerate(self.providers):
            start_time = time.perf_counter()
            try:
                logger.info("ai.chain_attempt", tier=i + 1, provider=name)
                resp = await provider.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Ensure structured data parsed if requested
                if response_schema and not resp.structured_data:
                    logger.warning(
                        "ai.chain_empty_structured_data",
                        provider=name,
                        raw_content=resp.content[:100],
                    )
                    # Let next provider try if structured parsing completely failed
                    continue

                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "ai.chain_success",
                    tier=i + 1,
                    provider=name,
                    model=resp.model,
                    latency_ms=round(latency_ms, 2),
                )
                return resp

            except Exception as exc:
                last_error = exc
                next_provider = self.providers[i + 1][0] if i + 1 < len(self.providers) else "none"
                logger.warning(
                    "ai.chain_failover",
                    failed_tier=i + 1,
                    failed_provider=name,
                    error=str(exc)[:150],
                    failing_over_to=next_provider,
                )

        raise last_error or LLMException("All providers in the fallback chain failed.")

    async def health_check(self) -> bool:
        """Return True if at least one provider in the chain is operational."""
        for name, provider in self.providers:
            try:
                if await provider.health_check():
                    return True
            except Exception:
                continue
        return False
