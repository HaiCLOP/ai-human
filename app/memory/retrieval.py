"""Memory retrieval and multi-factor ranking algorithms."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Sequence

from app.memory.models import ScoredMemory
from app.storage.repositories import MemoryRecord


def _token_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Fast lexical token overlap similarity."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def compute_memory_score(
    memory: MemoryRecord,
    query_text: str,
    semantic_similarity: float | None = None,
    now_utc: datetime | None = None,
    weight_semantic: float = 0.50,
    weight_recency: float = 0.30,
    weight_frequency: float = 0.20,
    decay_lambda: float = 0.05,
) -> float:
    """Calculate composite relevance score according to docs/MEMORY_DESIGN.md."""
    # 1. Semantic / Lexical Similarity
    if semantic_similarity is not None:
        s_sim = max(0.0, min(1.0, semantic_similarity))
    else:
        # Fallback to lexical token overlap
        s_sim = _token_jaccard_similarity(memory.statement, query_text)

    # 2. Recency Score with Exponential Decay
    now = now_utc or datetime.now(timezone.utc)
    try:
        last_accessed = datetime.fromisoformat(memory.last_accessed_at.replace("Z", "+00:00"))
        delta_days = max(0.0, (now - last_accessed).total_seconds() / 86400.0)
    except Exception:
        delta_days = 0.0

    r_recency = math.exp(-decay_lambda * delta_days)

    # 3. Frequency & Confidence
    f_frequency = min(1.0, memory.access_count / 5.0) * memory.confidence

    # Weighted Composite Score
    score = (weight_semantic * s_sim) + (weight_recency * r_recency) + (weight_frequency * f_frequency)
    return round(score, 4)


def rank_and_filter_memories(
    memories: Sequence[MemoryRecord],
    query_text: str,
    threshold: float = 0.50,
    top_k: int = 3,
    semantic_similarities: dict[str, float] | None = None,
) -> list[ScoredMemory]:
    """Score, filter by threshold, and rank memories."""
    similarities = semantic_similarities or {}
    scored: list[ScoredMemory] = []

    for mem in memories:
        sem_sim = similarities.get(mem.memory_id)
        score = compute_memory_score(mem, query_text, semantic_similarity=sem_sim)
        if score >= threshold:
            scored.append(
                ScoredMemory(
                    memory_id=mem.memory_id,
                    conversation_id=mem.conversation_id,
                    memory_type=mem.memory_type,
                    statement=mem.statement,
                    confidence=mem.confidence,
                    score=score,
                    access_count=mem.access_count,
                )
            )

    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:top_k]
