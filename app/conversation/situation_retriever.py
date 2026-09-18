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
        strategy_name: str | None = None,
        allow_question: bool = False,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve top-scored historical turns matching the conversational situation."""
        target_act = intent.social_act
        incoming_tokens = _tokenize(incoming_text)
        incoming_len = len(incoming_text.split())

        # 1. Fetch turns for this act
        act_turns = self.repo.get_turns(contact_id=contact_id, social_act=target_act, limit=30)
        global_act_turns = self.repo.get_turns(contact_id=None, social_act=target_act, limit=30)

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

        # 2. Multi-signal scoring (10 signals from Section 11)
        scored_turns: list[tuple[float, dict[str, Any]]] = []

        for turn in candidates.values():
            turn_act = turn.get("social_act", "other")
            turn_contact_id = turn.get("contact_id")
            turn_length_cat = turn.get("response_length_category", "short")
            turn_contact_text = turn.get("contact_text", "")
            turn_operator_text = turn.get("operator_text", "")
            turn_strategy = turn.get("response_strategy", "direct_answer")

            # 1. Social act match (0.25)
            m_act = 1.0 if turn_act == target_act else 0.0

            # 2. Relationship context match (0.15)
            m_rel = 1.0 if contact_id and turn_contact_id == contact_id else 0.5

            # 3. Strategy match (0.15)
            m_strat = 1.0 if strategy_name and turn_strategy.lower() == strategy_name.lower() else 0.5

            # 4. Effort & length category match (0.15)
            m_length = 1.0 if turn_length_cat == intent.expected_reply_length else 0.5

            # 5. Question structure alignment (0.10)
            has_q = "?" in turn_operator_text
            m_q = 1.0 if (has_q == allow_question) else 0.3

            # 6. Incoming message length proximity (0.05)
            turn_inc_len = len(turn_contact_text.split())
            len_diff = abs(incoming_len - turn_inc_len)
            m_inc_len = max(0.0, 1.0 - (len_diff / 10.0))

            # 7. Semantic / lexical similarity (0.10)
            turn_tokens = _tokenize(turn_contact_text)
            m_sem = _jaccard_similarity(incoming_tokens, turn_tokens)

            # 8. Topic keyword overlap (0.05)
            common_count = len(incoming_tokens & turn_tokens)
            m_topic = min(common_count / 3.0, 1.0)

            # Composite score
            if m_act < 0.5 and target_act != "other":
                score = (0.10 * m_rel + 0.10 * m_length + 0.10 * m_sem) * 0.5
            else:
                score = (
                    0.25 * m_act
                    + 0.15 * m_rel
                    + 0.15 * m_strat
                    + 0.15 * m_length
                    + 0.10 * m_q
                    + 0.05 * m_inc_len
                    + 0.10 * m_sem
                    + 0.05 * m_topic
                )

            scored_turns.append((score, turn))

        # Sort by score descending
        scored_turns.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, turn in scored_turns[:limit]:
            op_text = turn.get("operator_text", "")
            has_q = "?" in op_text
            results.append({
                "turn_id": turn["turn_id"],
                "contact_text": turn["contact_text"],
                "operator_text": op_text,
                "social_act": turn.get("social_act", "other"),
                "response_strategy": turn.get("response_strategy", "direct_answer"),
                "effort": turn.get("response_length_category", "short"),
                "has_question": has_q,
                "pattern": f"{turn.get('response_length_category', 'short')} response ({'with question' if has_q else 'no question'})",
                "score": round(score, 3),
            })

        logger.debug(
            "situation_retriever.retrieved",
            act=target_act,
            count=len(results),
            top_score=results[0]["score"] if results else 0.0,
        )
        return results

    @staticmethod
    def format_for_prompt(examples: list[dict[str, Any]]) -> str:
        """Format retrieved turns showing structure + example to prevent rote copying."""
        if not examples:
            return ""

        lines = ["[HISTORICAL BEHAVIORAL EXAMPLES — Learn structure, do NOT copy exact words]"]
        for idx, ex in enumerate(examples[:2], 1):
            act = ex.get("social_act", "other")
            strat = ex.get("response_strategy", "direct_reply")
            effort = ex.get("effort", "short")
            has_q = "true" if ex.get("has_question") else "false"
            pattern = ex.get("pattern", "short direct response")
            c_text = ex.get("contact_text", "")
            o_text = ex.get("operator_text", "")

            lines.append(f"Example {idx}:")
            lines.append(f"  SOCIAL ACT: {act}")
            lines.append(f"  STRATEGY: {strat}")
            lines.append(f"  EFFORT: {effort}")
            lines.append(f"  QUESTION: {has_q}")
            lines.append(f"  PATTERN: {pattern}")
            lines.append(f"  REAL CHAT: \"{c_text}\" → \"{o_text}\"")

        return "\n".join(lines)
