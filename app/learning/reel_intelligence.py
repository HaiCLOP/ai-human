"""Reel intelligence engine analyzing sharing habits, commentary, and anti-repetition memory."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from app.core.logging import get_logger
from app.learning.models import ReelIntelligenceProfile

logger = get_logger("learning.reels")


class ReelIntelligenceAnalyzer:
    """Extracts Reel sharing behaviors and prevents repetitive re-sharing of historical media."""

    def __init__(self):
        self._recently_shared_urls: set[str] = set()

    def analyze(self, all_messages: list[dict[str, Any]], operator_name: str) -> tuple[ReelIntelligenceProfile, list[dict[str, Any]]]:
        """Analyze reel sharing characteristics and extract individual reel records."""
        op_lower = operator_name.strip().lower()
        reel_messages = [m for m in all_messages if m.get("message_type") == "REEL"]

        reels_extracted: list[dict[str, Any]] = []
        sent_by_op = 0
        received_by_op = 0
        with_commentary = 0
        creators: list[str] = []
        reactions_on_reels = 0

        for idx, m in enumerate(reel_messages):
            url = m.get("shared_url") or ""
            text = m.get("text", "") or m.get("share_text", "") or ""
            owner = m.get("original_content_owner") or "unknown"
            sender = m.get("sender_name", "")
            is_op = 1 if sender.strip().lower() == op_lower else 0

            if is_op:
                sent_by_op += 1
            else:
                received_by_op += 1

            if text.strip():
                with_commentary += 1

            if owner and owner != "unknown":
                creators.append(owner)

            rx = m.get("reactions_json", "[]")
            rx_count = 0
            try:
                rx_list = json.loads(rx) if isinstance(rx, str) else rx
                rx_count = len(rx_list)
                if rx_count > 0:
                    reactions_on_reels += 1
            except Exception:
                pass

            reel_id = f"reel_{m.get('conversation_id', 'conv')}_{m.get('timestamp_ms', 0)}_{idx}"
            reels_extracted.append({
                "reel_id": reel_id,
                "conversation_id": m.get("conversation_id", ""),
                "contact_id": m.get("contact_id", ""),
                "sender_name": sender,
                "sender_is_operator": is_op,
                "reel_url": url,
                "share_text": text if text.strip() else None,
                "original_content_owner": owner,
                "timestamp_ms": m.get("timestamp_ms", 0),
                "reaction_count": rx_count,
                "topic_category": "meme" if any(w in text.lower() for w in ["us", "literally", "bhai", "bro", "relatable"]) else "general",
            })

        total_reels = len(reel_messages)
        commentary_ratio = round(with_commentary / total_reels, 3) if total_reels > 0 else 0.0
        reaction_rate = round(reactions_on_reels / total_reels, 3) if total_reels > 0 else 0.0
        top_creators = [c for c, _ in Counter(creators).most_common(5)]
        confidence = round(min(0.99, total_reels / (total_reels + 10)), 3)

        profile = ReelIntelligenceProfile(
            total_reels_shared=total_reels,
            sent_by_operator_count=sent_by_op,
            received_by_operator_count=received_by_op,
            shared_with_commentary_ratio=commentary_ratio,
            top_shared_creators=top_creators,
            recipient_reaction_rate=reaction_rate,
            confidence=confidence,
        )

        return profile, reels_extracted

    def is_reel_fresh(self, reel_url: str, historical_urls: set[str] | None = None) -> bool:
        """Verify that a reel has not been previously shared in conversation or during current session."""
        if not reel_url:
            return False
        if reel_url in self._recently_shared_urls:
            return False
        if historical_urls and reel_url in historical_urls:
            return False
        return True

    def mark_reel_shared(self, reel_url: str) -> None:
        """Register a reel URL into the anti-repetition memory."""
        self._recently_shared_urls.add(reel_url)
