"""OpenRouter AI Provider implementation using OpenAI-compatible HTTP chat completions."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from app.ai.provider import LLMProvider, LLMResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    LLMException,
    LLMRateLimitException,
    LLMResponseMalformedException,
    LLMServerException,
)
from app.core.logging import get_logger

logger = get_logger("ai.openrouter")

T = TypeVar("T", bound=BaseModel)


class OpenRouterProvider(LLMProvider):
    """Integrates with OpenRouter Cloud for access to free and open models."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        settings: Settings | None = None,
    ):
        cfg = settings or get_settings()
        self.api_key = api_key or cfg.OPENROUTER_API_KEY
        self.model = model or cfg.LLM_MODEL or "nvidia/nemotron-3-super-120b-a12b:free"
        self.fallback_model = fallback_model or cfg.LLM_FALLBACK_MODEL
        self.temperature = temperature if temperature is not None else cfg.LLM_TEMPERATURE
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else cfg.LLM_MAX_OUTPUT_TOKENS
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else cfg.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else cfg.LLM_MAX_RETRIES

        if not self.api_key:
            logger.warning("openrouter.api_key_not_set", message="OPENROUTER_API_KEY is empty.")

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Dispatch inference request to OpenRouter with automatic retries and fallback."""
        try:
            return await self._call_model(
                model_name=self.model,
                prompt=prompt,
                system_instruction=system_instruction,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if self.fallback_model and self.fallback_model != self.model:
                logger.warning(
                    "openrouter.primary_failed_failing_over",
                    primary=self.model,
                    fallback=self.fallback_model,
                    error=str(e),
                )
                return await self._call_model(
                    model_name=self.fallback_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            raise

    async def _call_model(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_output_tokens
        # Reasoning models (like Qwen 3.8 / Nemotron) consume tokens for internal thinking; ensure at least 800 tokens so JSON isn't truncated
        effective_tokens = max(tokens, 800) if ("qwen" in model_name.lower() or "free" in model_name.lower()) else tokens

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        user_content = prompt
        if response_schema:
            schema_json = json.dumps(response_schema.model_json_schema())
            user_content = f"{prompt}\n\nIMPORTANT: Respond with ONLY a valid JSON object matching this schema:\n{schema_json}"

        messages.append({"role": "user", "content": user_content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/instagram-ai-character",
            "X-Title": "Instagram AI Character",
        }

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": effective_tokens,
        }

        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(1, self.max_retries + 1):
                start_time = time.perf_counter()
                try:
                    logger.info("openrouter.generation_attempt", attempt=attempt, model=model_name)
                    response = await client.post(self.BASE_URL, headers=headers, json=payload)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Auto-fallback from upstream-limited :free tier to standard slug
                    if response.status_code in (404, 429) and ":free" in payload.get("model", ""):
                        fallback_slug = payload["model"].replace(":free", "")
                        logger.warning(
                            "openrouter.free_tier_unavailable_retrying_standard",
                            status=response.status_code,
                            original=payload["model"],
                            fallback=fallback_slug,
                        )
                        payload["model"] = fallback_slug
                        response = await client.post(self.BASE_URL, headers=headers, json=payload)

                    if response.status_code == 429:
                        wait = min(2.0 ** attempt, 8.0)
                        logger.warning("openrouter.rate_limited", attempt=attempt, status_code=429)
                        if attempt == self.max_retries:
                            raise LLMRateLimitException("OpenRouter 429: rate limit exceeded.", details={"status_code": 429})
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code >= 500:
                        wait = min(2.0 ** attempt, 6.0)
                        logger.warning("openrouter.server_error", status=response.status_code, attempt=attempt)
                        if attempt == self.max_retries:
                            raise LLMServerException(f"OpenRouter {response.status_code}: {response.text}")
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code != 200:
                        raise LLMException(f"OpenRouter HTTP {response.status_code}: {response.text}")


                    data = response.json()
                    raw_text = data["choices"][0]["message"]["content"] or ""

                    structured_data = None
                    if response_schema:
                        structured_data = self._parse_and_repair_json(raw_text)

                    usage = data.get("usage", {})
                    logger.info(
                        "openrouter.generation_success",
                        model=model_name,
                        latency_ms=round(latency_ms, 2),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                    )

                    return LLMResponse(
                        content=raw_text,
                        structured_data=structured_data,
                        model=model_name,
                        provider="openrouter",
                        latency_ms=latency_ms,
                        prompt_tokens=usage.get("prompt_tokens") or 0,
                        output_tokens=usage.get("completion_tokens") or 0,
                    )

                except (LLMRateLimitException, LLMServerException) as known_err:
                    last_error = known_err
                except Exception as exc:
                    last_error = exc
                    logger.warning("openrouter.attempt_failed", attempt=attempt, error=str(exc))
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(1.5 ** attempt, 5.0))

        raise last_error or LLMException("OpenRouter generation failed across all retries.")

    def _parse_and_repair_json(self, raw_text: str) -> dict[str, Any]:
        """Extract and repair JSON from potentially noisy or truncated model output."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        if not cleaned.startswith("{"):
            fb = cleaned.find("{")
            lb = cleaned.rfind("}")
            if fb != -1 and lb != -1 and lb > fb:
                cleaned = cleaned[fb : lb + 1].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Suffix completion repair
            for suffix in ['"}', '"}}', '"}', '}', ']}}']:
                try:
                    return json.loads(cleaned + suffix)
                except Exception:
                    continue

            # Fallback regex extraction of bubbles
            b_match = re.search(r'"bubbles"\s*:\s*(\[[^\]]+\])', cleaned)
            r_match = re.search(r'"reply_text"\s*:\s*"([^"]+)"', cleaned)
            if b_match:
                try:
                    bubbles = json.loads(b_match.group(1))
                    reply_text = r_match.group(1) if r_match else " ".join(bubbles)
                    return {
                        "intent": "REPLY",
                        "reply_text": reply_text,
                        "bubbles": bubbles,
                        "internal_reasoning": "Extracted via regex fallback",
                    }
                except Exception:
                    pass

            raise LLMResponseMalformedException(
                f"OpenRouter output was not valid JSON: {raw_text}",
                details={"raw_text": raw_text},
            )

    async def health_check(self) -> bool:
        """Verify API connectivity."""
        try:
            res = await self.generate("ping", max_tokens=2)
            return bool(res.text)
        except Exception:
            return False
