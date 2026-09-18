"""Topic saturation cache and repetition avoidance tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class TopicSaturationCache:
    """Sliding-window cache tracking opener n-grams and topic frequencies to prevent topic latching."""

    recent_character_messages: list[str] = field(default_factory=list)
    recent_openers: list[str] = field(default_factory=list)
    topic_frequency: dict[str, int] = field(default_factory=dict)

    TOPIC_KEYWORDS: dict[str, list[str]] = field(
        default_factory=lambda: {
            "sharma sir": ["sharma", "sharma sir", "sharma ji"],
            "tuition": ["tuition", "coaching", "batch"],
            "quadratic equations": ["quadratic", "quadratics", "equation", "equations"],
            "physics": ["physics", "verma sir", "friction", "laws of motion"],
            "exams": ["exam", "midterm", "unit test", "board", "pre-board"],
            "movies_entertainment": ["movie", "film", "cinema", "ok jaanu", "scene"],
            "music": ["song", "spotify", "music", "gaana", "playlist", "track"],
            "food": ["khana", "snack", "pizza", "burger", "biryani", "bhookh", "chai"],
            "gaming": ["game", "gaming", "bgmi", "valorant", "playstation"],
        }
    )

    OPENER_PATTERNS: list[str] = field(
        default_factory=lambda: [
            "arre yaar", "arre bhai", "sach bolu", "honestly", "bhai", "yaar",
            "ngl", "tbh", "wait", "actually", "sun", "dekho", "kuch nahi", "kch nhi"
        ]
    )

    def seed_from_history(self, messages: list[str]) -> None:
        """Seed recent messages into cache so it knows topic history immediately."""
        for msg in messages:
            self.register_character_message(msg)

    def register_character_message(self, text: str) -> None:
        """Record dispatched message and update saturation metrics."""
        clean = text.strip()
        if not clean:
            return

        self.recent_character_messages.append(clean)
        if len(self.recent_character_messages) > 10:
            self.recent_character_messages.pop(0)

        lower = clean.lower()

        # Track openers
        matched_opener = None
        for op in self.OPENER_PATTERNS:
            if lower.startswith(op):
                matched_opener = op
                break
        if matched_opener:
            self.recent_openers.append(matched_opener)
            if len(self.recent_openers) > 6:
                self.recent_openers.pop(0)

        # Update topic counts based on the last 5 messages
        self.topic_frequency.clear()
        for msg in self.recent_character_messages[-5:]:
            msg_lower = msg.lower()
            for topic_name, keywords in self.TOPIC_KEYWORDS.items():
                if any(kw in msg_lower for kw in keywords):
                    self.topic_frequency[topic_name] = self.topic_frequency.get(topic_name, 0) + 1

    def get_saturated_topics(self, threshold: int = 2) -> list[str]:
        """Return topics mentioned threshold or more times in the sliding window."""
        return [topic for topic, count in self.topic_frequency.items() if count >= threshold]

    def strip_repetitive_opener(self, text: str) -> str:
        """Strip opening phrase if it matches any of the last 2 messages' openers."""
        lower = text.lower().strip()
        for op in self.recent_openers[-2:]:
            if lower.startswith(op):
                stripped = text[len(op):].lstrip(" ,.-")
                return stripped if stripped else text
        return text
