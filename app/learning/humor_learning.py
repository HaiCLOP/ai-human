"""Humor, sarcasm, teasing, and playful banter learning engine."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.learning.models import HumorProfile

logger = get_logger("learning.humor")

TEASING_LEXICON = [
    "paka mat", "chup", "gadhe", "pagal", "kuch bhi", "sharam kar",
    "chal jhootha", "chal jhoothi", "overacting", "delulu", "dramebaaz",
]

DARK_HUMOR_LEXICON = [
    "dead inside", "marna hai", "fml", "rip", "mar jaunga", "mar jaungi",
    "khatam", "zinda hu bas", "suicidal thoughts", "brain dead", "existential",
]

LAUGH_MARKERS = [
    "😂", "😭", "💀", "haha", "hahaha", "lmao", "lol", "rofl", "hehe",
]


class HumorAnalyzer:
    """Analyzes humor frequencies, teasing patterns, and laughing reactions."""

    def analyze(self, contact_id: str, messages: list[dict[str, Any]]) -> HumorProfile:
        """Analyze humor dynamics across a collection of historical messages."""
        texts = [m.get("text", "").strip() for m in messages if m.get("text", "").strip()]
        total = len(texts)
        if total == 0:
            return HumorProfile(contact_id=contact_id, updated_at=datetime.now(timezone.utc).isoformat())

        laugh_count = 0
        sarcasm_count = 0
        teasing_count = 0
        dark_count = 0
        laugh_reactions = 0

        for m in messages:
            # Check reactions on message
            raw_rx = m.get("reactions_json", "[]")
            try:
                rx_list = json.loads(raw_rx) if isinstance(raw_rx, str) else raw_rx
                for r in rx_list:
                    emoji = r.get("reaction", "")
                    if any(l in emoji for l in ["😂", "😆", "🤣"]):
                        laugh_reactions += 1
            except Exception:
                pass

        for t in texts:
            lower = t.lower()
            if any(lm in lower for lm in LAUGH_MARKERS):
                laugh_count += 1
            if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in TEASING_LEXICON):
                teasing_count += 1
            if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in DARK_HUMOR_LEXICON):
                dark_count += 1
            if any(w in lower for w in ["sahi hai", "accha", "achha", "great", "kya baat", "wah"]):
                sarcasm_count += 1

        humor_frequency = round(min(1.0, (laugh_count + laugh_reactions) / total), 3)
        sarcasm_frequency = round(min(1.0, sarcasm_count / total), 3)
        teasing_frequency = round(min(1.0, teasing_count / total), 3)
        dark_humor_frequency = round(min(1.0, dark_count / total), 3)

        confidence = round(min(0.99, total / (total + 25)), 3)

        return HumorProfile(
            contact_id=contact_id,
            humor_frequency=humor_frequency,
            sarcasm_frequency=sarcasm_frequency,
            teasing_frequency=teasing_frequency,
            dark_humor_frequency=dark_humor_frequency,
            callback_frequency=round(teasing_frequency * 0.5, 3),
            laughing_reactions_count=laugh_reactions,
            confidence=confidence,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
