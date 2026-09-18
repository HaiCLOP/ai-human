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
    # ── English identity facts ──────────────────────────────────────────────
    (re.compile(r"\bi\s+live\s+in\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User lives in {0}"),
    (re.compile(r"\bi\s+work\s+as\s+an?\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User works as {0}"),
    (re.compile(r"\bi\s+am\s+an?\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User is {0}"),
    (re.compile(r"\bmy\s+name\s+is\s+([A-Za-z]+)", re.IGNORECASE), "FACT", "User's name is {0}"),
    (re.compile(r"\bi\s+have\s+a\s+(cat|dog|pet|car)\s+named\s+([A-Za-z]+)", re.IGNORECASE), "FACT", "User has a {0} named {1}"),
    (re.compile(r"\bmy\s+(brother|sister|bhai|didi|bhaiya)\s+(?:is\s+)?(?:named?\s+)?([A-Za-z]+)", re.IGNORECASE), "FACT", "User's sibling is {1}"),
    (re.compile(r"\bi(?:'m|\s+am)\s+(\d{1,2})\s*(?:years?\s*old|saal\s*ka|saal\s*ki)", re.IGNORECASE), "FACT", "User is {0} years old"),
    (re.compile(r"\bi\s+(?:study|padh)\s+(?:at|in)\s+([A-Za-z\s]+)", re.IGNORECASE), "FACT", "User studies at {0}"),
    (re.compile(r"\bi\s+(?:go|jata\s+hu|jaata\s+hu|jati\s+hu)\s+to\s+([A-Za-z\s]+)\s+(?:college|school|university)", re.IGNORECASE), "FACT", "User goes to {0}"),

    # ── English preferences (positive) ──────────────────────────────────────
    (re.compile(r"\bi\s+love\s+([A-Za-z\s]+)", re.IGNORECASE), "PREFERENCE", "User loves {0}"),
    (re.compile(r"\bi\s+(?:really\s+)?like\s+([A-Za-z\s]{3,30})", re.IGNORECASE), "PREFERENCE", "User likes {0}"),
    (re.compile(r"\bi\s+enjoy\s+([A-Za-z\s]{3,30})", re.IGNORECASE), "PREFERENCE", "User enjoys {0}"),
    (re.compile(r"\bmy\s+favourite\s+(?:is\s+)?([A-Za-z\s]+)", re.IGNORECASE), "PREFERENCE", "User's favourite is {0}"),
    (re.compile(r"\bi\s+watch\s+([A-Za-z\s]+)\s+(?:alot|a lot|daily|everyday)", re.IGNORECASE), "PREFERENCE", "User watches {0} regularly"),

    # ── English preferences (negative / avoidance) ───────────────────────────
    (re.compile(r"\bi\s+hate\s+([A-Za-z\s]+)", re.IGNORECASE), "PREFERENCE", "User hates {0}"),
    (re.compile(r"\bi\s+don'?t\s+(?:drink|smoke|do\s+drugs|do\s+nashe)", re.IGNORECASE), "PREFERENCE", "User does not drink/do drugs"),
    (re.compile(r"\bi\s+don'?t\s+(?:like|enjoy)\s+([A-Za-z\s]{3,25})", re.IGNORECASE), "PREFERENCE", "User dislikes {0}"),
    (re.compile(r"\bi\s+never\s+([A-Za-z\s]{3,25})", re.IGNORECASE), "PREFERENCE", "User never {0}"),

    # ── Hinglish identity facts ──────────────────────────────────────────────
    (re.compile(r"\bmera\s+naam\s+([A-Za-z\u0900-\u097F]+)\s+(?:hai|he)\b", re.IGNORECASE), "FACT", "User's name is {0}"),
    (re.compile(r"\bmai[n]?\s+([A-Za-z\u0900-\u097F]+)\s+(?:mein|me|mai)\s+rehta|rehti\s+hu", re.IGNORECASE), "FACT", "User lives in {0}"),
    (re.compile(r"\bmai[n]?\s+(\d{1,2})\s+(?:saal|year)\s+ka\b", re.IGNORECASE), "FACT", "User is {0} years old"),
    (re.compile(r"\bmai[n]?\s+(\d{1,2})\s+(?:saal|year)\s+ki\b", re.IGNORECASE), "FACT", "User is {0} years old"),
    (re.compile(r"\bmai[n]?\s+(?:class|standard|std)\s+(\d{1,2})\s+(?:mein|me|mai)\s+(?:hu|hun|hoon)", re.IGNORECASE), "FACT", "User is in class {0}"),

    # ── Hinglish preferences (positive) ─────────────────────────────────────
    (re.compile(r"\bmujhe\s+([A-Za-z\u0900-\u097F\s]{3,25})\s+(?:bahut\s+)?(?:pasand\s+hai|pasand\s+he|acha\s+lagta)", re.IGNORECASE), "PREFERENCE", "User likes {0}"),
    (re.compile(r"\bmai[n]?\s+([A-Za-z\s]{3,20})\s+(?:bahut\s+)?(?:sunта|sunta|sunta|sunta\s+hu)\b", re.IGNORECASE), "PREFERENCE", "User listens to {0}"),
    (re.compile(r"\bmai[n]?\s+([A-Za-z\s]{3,20})\s+(?:dekhta|dekhti)\s+(?:hu|hun|hoon)\b", re.IGNORECASE), "PREFERENCE", "User watches {0}"),
    (re.compile(r"\bmai[n]?\s+([A-Za-z\s]{3,20})\s+khelta\s+(?:hu|hun|hoon)\b", re.IGNORECASE), "PREFERENCE", "User plays {0}"),

    # ── Hinglish preferences (negative / avoidance) ──────────────────────────
    (re.compile(r"\bmai[n]?\s+nashe\s+(?:nahi|nhi|nahin)\s+(?:karta|karti|leta|leti)\b", re.IGNORECASE), "PREFERENCE", "User does not drink/do drugs"),
    (re.compile(r"\bmai[n]?\s+(?:sharab|daru|cigarette|cigg?|smoke|smok)\s+(?:nahi|nhi|nahin)\s+(?:pita|peeta|piti|peeti|karta|karti|leta|leti)\b", re.IGNORECASE), "PREFERENCE", "User does not drink/smoke"),
    (re.compile(r"\bmai[n]?\s+([A-Za-z\s]{3,20})\s+(?:nahi|nhi|nahin)\s+(?:karta|karti|khata|khati|sunta|sунта|dekhta|dekhti)\b", re.IGNORECASE), "PREFERENCE", "User does not {0}"),
    (re.compile(r"\bmujhe\s+([A-Za-z\u0900-\u097F\s]{3,25})\s+(?:pasand\s+nahi|pasand\s+nhi|bilkul\s+pasand\s+nahi|achha\s+nahi\s+lagta)", re.IGNORECASE), "PREFERENCE", "User dislikes {0}"),
    (re.compile(r"\bmujhe\s+([A-Za-z\u0900-\u097F\s]{3,25})\s+(?:se\s+)?nафrat\s+hai", re.IGNORECASE), "PREFERENCE", "User hates {0}"),

    # ── Habits & lifestyle ────────────────────────────────────────────────────
    (re.compile(r"\bmai[n]?\s+(?:roz|daily|har\s+roz|everyday)\s+([A-Za-z\s]{3,20})\s+(?:karta|karti|hu|hun)\b", re.IGNORECASE), "BEHAVIOR", "User does {0} daily"),
    (re.compile(r"\bmai[n]?\s+gym\s+(?:jata|jaata|jaati|jati)\s+(?:hu|hun|hoon)\b", re.IGNORECASE), "BEHAVIOR", "User goes to the gym"),
    (re.compile(r"\bmai[n]?\s+(?:raat\s+ko|late)\s+so\s+(?:ta|ti|ta\s+hu|ti\s+hu|jata|jaati)\b", re.IGNORECASE), "BEHAVIOR", "User sleeps late"),
    (re.compile(r"\bmai[n]?\s+(?:subah|morning\s+mein)\s+(?:jaldi|early)\s+(?:uthta|uthti)\b", re.IGNORECASE), "BEHAVIOR", "User wakes up early"),
    (re.compile(r"\bmai[n]?\s+gamer\s+(?:hu|hun|hoon)\b", re.IGNORECASE), "PREFERENCE", "User is a gamer"),
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
        top_k: int = 4,
    ) -> list[str]:
        """Fetch, score, filter, and touch candidate memories.

        Always prioritizes persistent identity facts (e.g. name) and preferences
        (e.g. likes, dislikes, habits), followed by semantically scored memories
        relevant to the immediate query.
        """
        all_memories = self.repo.get_memories_for_conversation(conversation_id)
        if not all_memories:
            return []

        statements: list[str] = []
        seen_statements: set[str] = set()

        # 1. Always prioritize persistent preferences and identity facts
        for m in all_memories:
            m_type = m.memory_type.upper()
            stmt_lower = m.statement.lower()
            is_priority = (
                m_type in ("PREFERENCE", "BEHAVIOR")
                or "name is" in stmt_lower
                or "lives in" in stmt_lower
                or "does not" in stmt_lower
                or "likes" in stmt_lower
                or "hates" in stmt_lower
            )
            if is_priority and stmt_lower not in seen_statements:
                statements.append(f"[{m.memory_type}] {m.statement}")
                seen_statements.add(stmt_lower)
                self.repo.touch_memory(m.memory_id)
                if len(statements) >= 3:
                    break

        # 2. Score remaining candidate memories for query relevance
        scored = rank_and_filter_memories(
            memories=all_memories,
            query_text=current_query,
            threshold=threshold,
            top_k=top_k,
        )

        for sm in scored:
            stmt_str = f"[{sm.memory_type}] {sm.statement}"
            if sm.statement.lower() not in seen_statements:
                statements.append(stmt_str)
                seen_statements.add(sm.statement.lower())
                self.repo.touch_memory(sm.memory_id)
                if len(statements) >= 5:
                    break

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
