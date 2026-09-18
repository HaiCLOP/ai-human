"""Memory data structures and callback models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MemoryType = Literal["FACT", "PREFERENCE", "RUNNING_JOKE", "TOPIC"]


@dataclass
class ScoredMemory:
    """Memory record with composite relevance score."""

    memory_id: str
    conversation_id: str
    memory_type: str
    statement: str
    confidence: float
    score: float
    access_count: int


@dataclass
class RunningJokeCallback:
    """Shared running joke or ongoing conversational callback."""

    callback_id: str
    trigger_keywords: list[str]
    description: str
    punchline_hint: str
    last_used: str
    use_count: int
