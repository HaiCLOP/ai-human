"""Abstract LLM provider interface and response data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Standardized response container across all AI providers."""

    content: str
    structured_data: dict[str, Any] | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


class LLMProvider(ABC):
    """Abstract interface that all LLM implementations must satisfy."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Execute text generation or structured generation asynchronously."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify API connectivity and credential validity."""
        pass
