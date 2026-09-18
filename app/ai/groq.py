"""Groq Cloud LLM provider implementation with retry backoff and structured output support."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from app.ai.provider import LLMProvider, LLMResponse
from app.core.config import get_settings
from app.core.exceptions import (
    LLMException,
    LLMQuotaExhaustedException,
    LLMRateLimitException,
    LLMResponseMalformedException,
    LLMTimeoutException,
)
from app.core.logging import get_logger

logger = get_logger("ai.groq")
T = TypeVar("T", bound=BaseModel)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def _normalize_schema_for_groq(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure schema complies with Groq / OpenAI strict json_schema requirements."""
    res = dict(schema)
    if "properties" in res:
        res["additionalProperties"] = False
        all_props = list(res["properties"].keys())
        req = set(res.get("required", [])) | set(all_props)
        res["required"] = sorted(list(req))
        normalized_props = {}
        for k, v in res["properties"].items():
            if isinstance(v, dict):
                normalized_props[k] = _normalize_schema_for_groq(v)
            else:
                normalized_props[k] = v
        res["properties"] = normalized_props
    return res


class GroqProvider(LLMProvider):
    """Concrete provider for Groq Cloud API offering ultrafast inference and high free limits."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "GROQ_API_KEY", "")
        self.model = model or settings.LLM_MODEL
        self.fallback_model = fallback_model or settings.LLM_FALLBACK_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else settings.LLM_MAX_OUTPUT_TOKENS
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Execute chat completion on Groq with fallback support."""
        try:
            return await self._generate_with_model(
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
                    "groq.primary_failed_failing_over",
                    primary_model=self.model,
                    fallback_model=self.fallback_model,
                    error=str(e),
                )
                return await self._generate_with_model(
                    model_name=self.fallback_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            raise

    async def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMException("GROQ_API_KEY is not configured.")

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_output_tokens

        messages: list[dict[str, str]] = []
        sys_content = system_instruction or ""

        if response_schema:
            schema_json = json.dumps(response_schema.model_json_schema())
            schema_directive = (
                f"\n\nCRITICAL: You MUST respond ONLY with a valid JSON object strictly matching this schema:\n"
                f"{schema_json}\n"
                f"Do not wrap your output in markdown backticks or commentary. Output raw valid JSON only."
            )
            sys_content = (sys_content + schema_directive).strip()

        if sys_content:
            messages.append({"role": "system", "content": sys_content})

        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }

        if response_schema:
            try:
                strict_schema = _normalize_schema_for_groq(response_schema.model_json_schema())
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "character_response_plan",
                        "schema": strict_schema,
                        "strict": True,
                    },
                }
            except Exception:
                payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(1, self.max_retries + 1):
                start_time = time.perf_counter()
                try:
                    logger.info("groq.generation_attempt", attempt=attempt, model=model_name)
                    response = await client.post(GROQ_CHAT_URL, headers=headers, json=payload)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    if response.status_code == 200:
                        data = response.json()
                        raw_text = data["choices"][0]["message"]["content"] or ""
                        structured_data = None

                        if response_schema:
                            cleaned = raw_text.strip()
                            if "```" in cleaned:
                                fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
                                if fence_match:
                                    cleaned = fence_match.group(1).strip()
                            if not cleaned.startswith("{"):
                                fb = cleaned.find("{")
                                lb = cleaned.rfind("}")
                                if fb != -1 and lb != -1 and lb > fb:
                                    cleaned = cleaned[fb : lb + 1].strip()

                            try:
                                structured_data = json.loads(cleaned)
                            except json.JSONDecodeError as jde:
                                # Attempt suffix repair
                                repaired = None
                                for suffix in ['"}', '"}}', '"}', '}', ']}}']:
                                    try:
                                        repaired = json.loads(cleaned + suffix)
                                        break
                                    except Exception:
                                        continue
                                if repaired:
                                    structured_data = repaired
                                else:
                                    # Attempt regex fallback
                                    b_match = re.search(r'"bubbles"\s*:\s*(\[[^\]]+\])', cleaned)
                                    r_match = re.search(r'"reply_text"\s*:\s*"([^"]+)"', cleaned)
                                    if b_match:
                                        try:
                                            bubbles = json.loads(b_match.group(1))
                                            reply_text = r_match.group(1) if r_match else " ".join(bubbles)
                                            structured_data = {
                                                "intent": "REPLY",
                                                "reply_text": reply_text,
                                                "bubbles": bubbles,
                                                "internal_reasoning": "Extracted via regex fallback",
                                            }
                                        except Exception:
                                            pass

                                if not structured_data:
                                    logger.warning("groq.json_decode_error", raw_text=raw_text)
                                    raise LLMResponseMalformedException(
                                        f"Groq output was not valid JSON: {raw_text}",
                                        details={"raw_text": raw_text},
                                    ) from jde

                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                        logger.info(
                            "groq.generation_success",
                            model=model_name,
                            latency_ms=round(latency_ms, 2),
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                        )

                        return LLMResponse(
                            content=raw_text,
                            structured_data=structured_data,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            latency_ms=latency_ms,
                            model=model_name,
                            provider="groq",
                        )

                    elif response.status_code == 429:
                        last_error = LLMRateLimitException(f"Rate limited by Groq: {response.text}")
                        logger.warning("groq.rate_limited", status_code=429, error=response.text)
                    elif response.status_code >= 500:
                        last_error = LLMException(f"Groq server error {response.status_code}: {response.text}")
                        logger.warning("groq.server_error", status_code=response.status_code, error=response.text)
                    elif response.status_code == 400 and payload.get("response_format", {}).get("type") == "json_schema":
                        logger.warning("groq.json_schema_unsupported_fallback_to_json_object", model=model_name)
                        payload["response_format"] = {"type": "json_object"}
                        continue
                    else:
                        raise LLMException(f"Groq API error {response.status_code}: {response.text}")

                except httpx.TimeoutException:
                    last_error = LLMTimeoutException(f"Groq call timed out after {self.timeout_seconds}s.")
                    logger.warning("groq.timeout", model=model_name, attempt=attempt)
                except (LLMResponseMalformedException, LLMException):
                    raise
                except Exception as ex:
                    last_error = LLMException(f"Groq request failed: {ex}")
                    logger.warning("groq.request_failed", model=model_name, attempt=attempt, error=str(ex))

                # Backoff
                if attempt < self.max_retries:
                    backoff = min(15.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
                    logger.info("groq.backing_off", sleep_seconds=round(backoff, 2))
                    await asyncio.sleep(backoff)

        raise last_error or LLMException(f"Groq generation failed on {model_name} after {self.max_retries} attempts.")

    async def health_check(self) -> bool:
        """Verify Groq API connectivity and credential validity."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return res.status_code == 200
        except Exception as e:
            logger.warning("groq.health_check_failed", error=str(e))
            return False
