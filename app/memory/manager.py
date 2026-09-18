"""Memory manager orchestrating retrieval, scoring, and asynchronous fact extraction."""

from __future__ import annotations

import re
from typing import Sequence

from app.core.logging import get_logger
from app.memory.retrieval import rank_and_filter_memories
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import MemoryRecord, MemoryRepository

logger = get_logger("memory.manager")

# Rule-based heuristics for fast, zero-cost fact and preference extraction
FACT_EXTRACTION_PATTERNS = [
    (re.compile(r"\bi\s+live\s+in\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User lives in {0}"),
    (re.compile(r"\bi\s+work\s+as\s+an?\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User works as {0}"),
    (re.compile(r"\bi\s+am\s+an?\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User is {0}"),
    (re.compile(r"\bi\s+hate\s+([A-Za-z\s]+)", re.IGNORECASE), "PREFERENCE", "User hates {0}"),
    (re.compile(r"\bi\s+love\s+([A-Za-z\s]+)", re.IGNORECASE), "PREFERENCE", "User loves {0}"),
    (re.compile(r"\bmy\s+name\s+is\s+([A-Za-z]+)", re.IGNORECASE), "FACT", "User's name is {0}"),
    (re.compile(r"\bi\s+have\s+a\s+(cat|dog|pet|car)\s+named\s+([A-Za-z]+)", re.IGNORECASE), "FACT", "User has a {0} named {1}"),
]


class MemoryManager:
    """Orchestrates short-term, long-term factual, and preference memories."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.repo = MemoryRepository(self.db)

    def get_relevant_memories(
        self,
        conversation_id: str,
        current_query: str,
        threshold: float = 0.40,
        top_k: int = 3,
    ) -> list[str]:
        """Fetch, score, filter, and touch candidate memories."""
        all_memories = self.repo.get_memories_for_conversation(conversation_id)
        if not all_memories:
            return []

        scored = rank_and_filter_memories(
            memories=all_memories,
            query_text=current_query,
            threshold=threshold,
            top_k=top_k,
        )

        statements: list[str] = []
        for sm in scored:
            statements.append(f"[{sm.memory_type}] {sm.statement}")
            self.repo.touch_memory(sm.memory_id)

        logger.info(
            "memory.retrieval_complete",
            conversation_id=conversation_id,
            matched_count=len(statements),
        )
        return statements

    def extract_and_persist_facts(
        self,
        conversation_id: str,
        user_message: str,
    ) -> list[MemoryRecord]:
        """Asynchronously extract and persist facts/preferences from inbound user message."""
        existing_memories = self.repo.get_memories_for_conversation(conversation_id)
        existing_statements = {m.statement.lower() for m in existing_memories}

        extracted: list[MemoryRecord] = []

        for pattern, mem_type, template in FACT_EXTRACTION_PATTERNS:
            match = pattern.search(user_message)
            if match:
                groups = [g.strip() for g in match.groups()]
                statement = template.format(*groups).strip()

                # Avoid duplicate insertion
                if statement.lower() in existing_statements:
                    continue

                rec = self.repo.add_memory(
                    conversation_id=conversation_id,
                    memory_type=mem_type,
                    statement=statement,
                    confidence=0.85,
                )
                existing_statements.add(statement.lower())
                extracted.append(rec)
                logger.info(
                    "memory.fact_extracted",
                    conversation_id=conversation_id,
                    memory_type=mem_type,
                    statement=statement,
                )

        return extracted
