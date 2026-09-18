"""Situation-Aware Historical Pattern and Turn Retriever.

Retrieves historical conversation turns matching current social intent,
relationship context, and effort parity.
"""

from __future__ import annotations

import re
from typing import Any
from app.conversation.social_intent import SocialIntent
from app.core.logging import get_logger
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.historical_repo import HistoricalRepository

logger = get_logger("conversation.situation_retriever")


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric word set."""
    return set(re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class SituationAwareRetriever:
    """Retrieves and scores historical conversation turns for in-context few-shot guidance."""

    def __init__(self, db: DatabaseManager | None = None, repo: HistoricalRepository | None = None):
        self.db = db or get_db_manager()
        self.repo = repo or HistoricalRepository(self.db)

    def retrieve_examples(
        self,
        incoming_text: str,
        intent: SocialIntent,
        contact_id: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve top-scored historical turns matching the conversational situation."""
        target_act = intent.social_act
        incoming_tokens = _tokenize(incoming_text)

        # 1. Fetch turns for this act
        act_turns = self.repo.get_turns(contact_id=contact_id, social_act=target_act, limit=25)
        global_act_turns = self.repo.get_turns(contact_id=None, social_act=target_act, limit=25)

        # Merge unique turns by turn_id
        candidates: dict[str, dict[str, Any]] = {}
        for t in act_turns + global_act_turns:
            candidates[t["turn_id"]] = t

        # If not enough candidates, fetch contact's general turns
        if len(candidates) < limit and contact_id:
            general_turns = self.repo.get_turns(contact_id=contact_id, social_act=None, limit=20)
            for t in general_turns:
                if t["turn_id"] not in candidates:
                    candidates[t["turn_id"]] = t

        if not candidates:
            return []

        # 2. Multi-signal scoring
        scored_turns: list[tuple[float, dict[str, Any]]] = []

        for turn in candidates.values():
            turn_act = turn.get("social_act", "other")
            turn_contact_id = turn.get("contact_id")
            turn_length_cat = turn.get("response_length_category", "short")
            turn_contact_text = turn.get("contact_text", "")

            # M_act (0.30)
            m_act = 1.0 if turn_act == target_act else 0.0

            # M_rel (0.25)
            m_rel = 1.0 if contact_id and turn_contact_id == contact_id else 0.5

            # M_length (0.20)
            m_length = 1.0 if turn_length_cat == intent.expected_reply_length else 0.5

            # M_sem (0.15)
            turn_tokens = _tokenize(turn_contact_text)
            m_sem = _jaccard_similarity(incoming_tokens, turn_tokens)

            # M_topic (0.10)
            common_count = len(incoming_tokens & turn_tokens)
            m_topic = min(common_count / 3.0, 1.0)

            # Strict guard: semantic overlap CANNOT override speech-act mismatch
            if m_act < 0.5 and target_act != "other":
                score = (0.10 * m_rel + 0.10 * m_length + 0.10 * m_sem) * 0.5
            else:
                score = (
                    0.30 * m_act
                    + 0.25 * m_rel
                    + 0.20 * m_length
                    + 0.15 * m_sem
                    + 0.10 * m_topic
                )

            scored_turns.append((score, turn))

        # Sort by score descending
        scored_turns.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, turn in scored_turns[:limit]:
            results.append({
                "turn_id": turn["turn_id"],
                "contact_text": turn["contact_text"],
                "operator_text": turn["operator_text"],
                "social_act": turn["social_act"],
                "response_strategy": turn.get("response_strategy", "direct_answer"),
                "score": round(score, 3),
            })

        logger.debug(
            "situation_retriever.retrieved",
            act=target_act,
            count=len(results),
            top_score=results[0]["score"] if results else 0.0,
        )
        return results
