"""Memory and recurring topic extraction from historical conversation exchanges."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.learning.models import MemoryCandidate

logger = get_logger("learning.memory")

# Key phrase patterns that indicate personal facts, interests, plans, or running topics
PREFERENCE_PATTERNS = [
    (re.compile(r"\b(?:i\s+like|i\s+love|mujhe\s+pasand|fav|favorite)\s+([a-zA-Z0-9\s]{3,25})", re.IGNORECASE), "preference"),
    (re.compile(r"\b(?:hate|mujhe\s+bekaar|dislike|paka\s+diya)\s+([a-zA-Z0-9\s]{3,25})", re.IGNORECASE), "preference"),
    (re.compile(r"\b(?:watching|dekh\s+raha|dekh\s+rahi|binge)\s+([a-zA-Z0-9\s]{3,25})", re.IGNORECASE), "interest"),
    (re.compile(r"\b(?:kal|tomorrow|weekend|parso|aaj\s+raat)\s+([a-zA-Z0-9\s]{3,30})", re.IGNORECASE), "plan"),
    (re.compile(r"\b(?:tuition|coaching|sharma\s+sir|verma\s+sir|school|college)\b", re.IGNORECASE), "relationship_context"),
]

TOPIC_KEYWORDS = [
    "movie", "series", "anime", "football", "cricket", "gaming", "valorant",
    "gym", "biryani", "momo", "pizza", "coffee", "exam", "assignment", "trip",
    "pvr", "hostel", "songs", "playlist", "bday", "birthday",
]


class MemoryExtractor:
    """Extracts high-value, recurring memory candidates and topics from chat history."""

    @staticmethod
    def _generate_memory_id(contact_id: str, statement: str) -> str:
        clean = f"{contact_id}_{statement.strip().lower()}"
        return f"mem_{hashlib.sha256(clean.encode('utf-8')).hexdigest()[:12]}"

    def extract(
        self,
        contact_id: str,
        messages: list[dict[str, Any]],
        min_confidence: float = 0.55,
    ) -> tuple[list[MemoryCandidate], list[dict[str, Any]]]:
        """Extract memory candidates and recurring topics from message history."""
        texts_with_ts = [
            (m.get("text", "").strip(), m.get("timestamp_ms", 0), m.get("sender_name", ""))
            for m in messages
            if m.get("text", "").strip()
        ]

        if not texts_with_ts:
            return [], []

        candidates: list[MemoryCandidate] = []
        topic_counts: dict[str, int] = Counter()
        topic_last_seen: dict[str, int] = {}

        # 1. Topic frequency scan
        for text, ts, _ in texts_with_ts:
            lower = text.lower()
            for kw in TOPIC_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    topic_counts[kw] += 1
                    if kw not in topic_last_seen or ts > topic_last_seen[kw]:
                        topic_last_seen[kw] = ts

        # 2. Preference and fact extraction
        for text, ts, sender in texts_with_ts:
            for pattern, category in PREFERENCE_PATTERNS:
                match = pattern.search(text)
                if match:
                    statement = f"{sender}: {text}"
                    if len(statement) > 120:
                        statement = statement[:117] + "..."

                    mem_id = self._generate_memory_id(contact_id, text[:60])
                    dt_str = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d") if ts > 0 else ""

                    candidates.append(
                        MemoryCandidate(
                            memory_id=mem_id,
                            contact_id=contact_id,
                            category=category,
                            statement=statement,
                            importance=0.70 if category in ["preference", "plan"] else 0.50,
                            confidence=0.75,
                            frequency=1,
                            recency=dt_str,
                            source_count=1,
                        )
                    )

        # 3. Add recurring topics with count >= 2 as recurring_topic memories
        for topic, count in topic_counts.items():
            if count >= 2:
                ts = topic_last_seen.get(topic, 0)
                dt_str = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d") if ts > 0 else ""
                stmt = f"Frequent mutual conversation topic: {topic} (discussed {count} times)"
                mem_id = self._generate_memory_id(contact_id, f"topic_{topic}")

                importance = min(0.90, 0.50 + (count * 0.05))
                confidence = min(0.95, 0.60 + (count * 0.08))

                candidates.append(
                    MemoryCandidate(
                        memory_id=mem_id,
                        contact_id=contact_id,
                        category="recurring_topic",
                        statement=stmt,
                        importance=round(importance, 2),
                        confidence=round(confidence, 2),
                        frequency=count,
                        recency=dt_str,
                        source_count=count,
                    )
                )

        # Deduplicate candidates by memory_id
        unique_candidates: dict[str, MemoryCandidate] = {}
        for c in candidates:
            if c.memory_id not in unique_candidates:
                unique_candidates[c.memory_id] = c
            else:
                existing = unique_candidates[c.memory_id]
                existing.frequency += 1
                existing.source_count += 1
                existing.confidence = min(0.99, existing.confidence + 0.05)

        filtered_candidates = [
            c for c in unique_candidates.values() if c.confidence >= min_confidence
        ]

        topics_data = [
            {
                "topic_id": f"top_{contact_id}_{kw}",
                "contact_id": contact_id,
                "topic_name": kw,
                "mention_count": count,
                "last_mentioned_at": datetime.fromtimestamp(
                    topic_last_seen.get(kw, 0) / 1000.0, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
            }
            for kw, count in topic_counts.items()
            if count >= 2
        ]

        return filtered_candidates, topics_data
