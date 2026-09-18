"""Independent contact style analyzer ensuring strict separation between conversation partners."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.learning.models import ContactStyleProfile
from app.learning.operator_style import EMOJI_PATTERN, SLANG_LEXICON

logger = get_logger("learning.contact_style")

SARCASM_MARKERS = [
    "sahi hai", "accha", "achha", "haan haan", "wah", "great", "kya baat",
    "bilkul", "obviously", "sure", "nice", "dead", "rip", "lmao", "lol",
]

FORMAL_MARKERS = [
    "please", "kindly", "thanks", "thank you", "hello", "hi sir", "regards",
    "dear", "apologize", "sorry for", "appreciate",
]


class ContactStyleAnalyzer:
    """Analyzes message patterns of individual contacts without cross-contact contamination."""

    @staticmethod
    def calculate_brevity(avg_chars: float) -> str:
        """Classify message brevity into human-readable tier."""
        if avg_chars <= 22.0:
            return "high"
        elif avg_chars <= 65.0:
            return "medium"
        return "low"

    def analyze(self, contact_id: str, display_name: str, contact_messages: list[dict[str, Any]]) -> ContactStyleProfile:
        """Generate an independent communication profile for a specific contact."""
        texts = [m.get("text", "").strip() for m in contact_messages if m.get("text", "").strip()]
        total = len(texts)

        if total == 0:
            return ContactStyleProfile(
                contact_id=contact_id,
                display_name=display_name,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

        # 1. Message Length & Brevity
        char_lengths = [len(t) for t in texts]
        avg_len = round(statistics.mean(char_lengths), 1)
        brevity = self.calculate_brevity(avg_len)

        # 2. Emojis
        all_emojis: list[str] = []
        msg_with_emojis = 0
        for t in texts:
            found = EMOJI_PATTERN.findall(t)
            if found:
                msg_with_emojis += 1
                for chunk in found:
                    for c in chunk:
                        all_emojis.append(c)

        emoji_density = round(msg_with_emojis / total, 3)
        top_emojis = Counter(all_emojis).most_common(5)

        # 3. Hinglish & Slang
        slang_used: set[str] = set()
        hinglish_count = 0
        for t in texts:
            lower = t.lower()
            matched = False
            for s in SLANG_LEXICON:
                if re.search(rf"\b{re.escape(s)}\b", lower):
                    slang_used.add(s)
                    matched = True
            if matched:
                hinglish_count += 1
        hinglish_ratio = round(hinglish_count / total, 3)

        # 4. Sarcasm Score
        sarcasm_hits = sum(
            1 for t in texts if any(re.search(rf"\b{re.escape(m)}\b", t.lower()) for m in SARCASM_MARKERS)
        )
        sarcasm_score = round(min(1.0, (sarcasm_hits / total) * 2.5), 3)

        # 5. Formality Score
        formal_hits = sum(
            1 for t in texts if any(re.search(rf"\b{re.escape(f)}\b", t.lower()) for f in FORMAL_MARKERS)
        )
        # Trailing periods on single lines also contribute to formality
        period_count = sum(1 for t in texts if t.endswith(".") and not t.endswith(".."))
        formality_score = round(min(1.0, (formal_hits * 2 + period_count) / (total * 1.5)), 3)

        # Confidence based on contact message volume
        confidence = round(min(0.99, total / (total + 15)), 3)

        return ContactStyleProfile(
            contact_id=contact_id,
            display_name=display_name,
            total_messages=total,
            brevity_level=brevity,
            avg_message_length_chars=avg_len,
            hinglish_ratio=hinglish_ratio,
            emoji_density=emoji_density,
            sarcasm_score=sarcasm_score,
            formality_score=formality_score,
            favorite_emojis=top_emojis,
            slang_terms_used=sorted(list(slang_used))[:15],
            confidence=confidence,
            evidence_count=total,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
