"""Google Gemini LLM provider implementation with retry backoff and structured output support."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, TypeVar

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

logger = get_logger("ai.gemini")
T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """Concrete provider for Google Gemini API."""

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
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.LLM_MODEL
        self.fallback_model = fallback_model or settings.LLM_FALLBACK_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else settings.LLM_MAX_OUTPUT_TOKENS
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES


        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                logger.warning("gemini.api_key_not_set", message="GEMINI_API_KEY is empty. Real generation calls will fail.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("gemini.client_init_failed", error=str(e))
                raise LLMException(f"Failed to initialize Gemini client: {e}") from e
        return self._client

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_output_tokens

        config_dict: dict[str, Any] = {
            "temperature": temp,
            "max_output_tokens": tokens,
        }
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        if response_schema:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_schema"] = response_schema

        try:
            return await self._generate_with_model(
                model_name=self.model,
                prompt=prompt,
                config_dict=config_dict,
                response_schema=response_schema,
            )
        except Exception as e:
            if self.fallback_model and self.fallback_model != self.model:
                logger.warning(
                    "gemini.primary_failed_failing_over",
                    primary_model=self.model,
                    fallback_model=self.fallback_model,
                    error=str(e),
                )
                return await self._generate_with_model(
                    model_name=self.fallback_model,
                    prompt=prompt,
                    config_dict=config_dict,
                    response_schema=response_schema,
                )
            raise

    async def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        config_dict: dict[str, Any],
        response_schema: type[T] | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                logger.info(
                    "gemini.generation_attempt",
                    attempt=attempt,
                    model=model_name,
                )

                # Run synchronous client SDK call in async threadpool to keep event loop responsive
                def _call():
                    from google.genai import types
                    config = types.GenerateContentConfig(**config_dict)
                    return client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(_call),
                    timeout=self.timeout_seconds,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                raw_text = response.text or ""

                structured_data = None
                if response_schema:
                    cleaned_json = raw_text.strip()
                    # Strip code fences if present
                    if "```" in cleaned_json:
                        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_json)
                        if fence_match:
                            cleaned_json = fence_match.group(1).strip()
                    # If still not starting with {, search for first { and last }
                    if not cleaned_json.startswith("{"):
                        fb = cleaned_json.find("{")
                        lb = cleaned_json.rfind("}")
                        if fb != -1 and lb != -1 and lb > fb:
                            cleaned_json = cleaned_json[fb : lb + 1].strip()

                    try:
                        structured_data = json.loads(cleaned_json)
                    except json.JSONDecodeError as json_err:
                        # Attempt suffix repair
                        repaired = None
                        for suffix in ['"}', '"}}', '"}', '}', ']}}']:
                            try:
                                repaired = json.loads(cleaned_json + suffix)
                                break
                            except Exception:
                                continue
                        if repaired:
                            structured_data = repaired
                        else:
                            # Attempt regex fallback
                            b_match = re.search(r'"bubbles"\s*:\s*(\[[^\]]+\])', cleaned_json)
                            r_match = re.search(r'"reply_text"\s*:\s*"([^"]+)"', cleaned_json)
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
                            logger.warning("gemini.json_decode_error", raw_text=raw_text)
                            raise LLMResponseMalformedException(
                                f"Gemini output was not valid JSON: {raw_text}",
                                details={"raw_text": raw_text},
                            ) from json_err

                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

                logger.info(
                    "gemini.generation_success",
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
                    provider="gemini",
                )

            except asyncio.TimeoutError as te:
                last_error = LLMTimeoutException(f"Gemini call timed out after {self.timeout_seconds}s.")
                logger.warning("gemini.timeout", model=model_name, attempt=attempt, timeout_seconds=self.timeout_seconds)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str:
                    last_error = LLMRateLimitException(f"Rate limited by Gemini (attempt {attempt}): {e}")
                elif "quota" in err_str:
                    last_error = LLMQuotaExhaustedException(f"Gemini quota exhausted: {e}")
                else:
                    last_error = LLMException(f"Gemini API call failed: {e}")

                logger.warning("gemini.attempt_failed", model=model_name, attempt=attempt, error=str(e))

                # If the error is 503 high demand, fail over immediately rather than burning retries
                if "503" in err_str or "high demand" in err_str or "unavailable" in err_str:
                    logger.warning("gemini.model_unavailable_high_demand", model=model_name)
                    break

            # Exponential backoff with jitter
            if attempt < self.max_retries:
                backoff = min(30.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0.1, 1.0)
                logger.info("gemini.backing_off", sleep_seconds=round(backoff, 2))
                await asyncio.sleep(backoff)

        raise last_error or LLMException(f"Gemini generation failed on {model_name} after {self.max_retries} attempts.")

    async def health_check(self) -> bool:
        """Verify client configuration and perform a minimal ping."""
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            # Perform a minimal generation test
            def _ping():
                return client.models.generate_content(
                    model=self.model,
                    contents="ping",
                )

            res = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=10.0)
            return bool(res and res.text)
        except Exception as e:
            logger.warning("gemini.health_check_failed", error=str(e))
            return False
