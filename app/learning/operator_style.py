"""Global operator communication style analyzer with time-aware weighting and confidence scoring."""

from __future__ import annotations

import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.learning.models import OperatorStyleProfile

logger = get_logger("learning.operator_style")

# Regex pattern for emojis
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f\u200d]+",
    flags=re.UNICODE,
)

# Common Indian / Gen Z slang markers
SLANG_LEXICON = [
    "bro", "yaar", "bhai", "scene", "scenes", "fml", "ngl", "fr", "dead",
    "rn", "tbh", "idk", "bc", "bruh", "wbu", "kuch nahi", "sahi", "sahi hai",
    "chalega", "paka mat", "dimag kharab", "mast", "accha", "achha", "theek",
    "sorted", "delulu", "cooked", "lmao", "lol", "badiya", "kya", "nahi",
    "haan", "dekh", "bol", "chal", "matlab", "thoda", "pata", "chalo",
]


class OperatorStyleAnalyzer:
    """Extracts linguistic, formatting, rhythm, and emoji habits from the operator's message history."""

    def __init__(
        self,
        recent_window_days: int = 90,
        long_term_weight: float = 0.60,
        recent_weight: float = 0.40,
    ):
        self.recent_window_days = recent_window_days
        self.long_term_weight = long_term_weight
        self.recent_weight = recent_weight

    @staticmethod
    def calculate_confidence(evidence_count: int, scale: int = 30) -> float:
        """Statistical confidence score asymptotically approaching 0.99 with evidence volume."""
        if evidence_count <= 0:
            return 0.0
        return round(min(0.99, evidence_count / (evidence_count + scale)), 3)

    def analyze_subset(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze a specific collection of operator messages."""
        if not messages:
            return {
                "total": 0,
                "lowercase_ratio": 0.0,
                "ending_period_ratio": 0.0,
                "contraction_apostrophe_ratio": 0.0,
                "avg_length_chars": 0.0,
                "median_length_chars": 0.0,
                "avg_words": 0.0,
                "emoji_density": 0.0,
                "hinglish_ratio": 0.0,
                "question_frequency": 0.0,
                "burst_ratio": 0.0,
                "slang_frequencies": {},
                "top_emojis": [],
            }

        texts = [m.get("text", "").strip() for m in messages if m.get("text", "").strip()]
        total = len(texts)
        if total == 0:
            return {"total": 0}

        # 1. Formatting
        lowercase_count = sum(1 for t in texts if t.islower() or (len(t) > 0 and t[0].islower()))
        lowercase_ratio = round(lowercase_count / total, 3)

        ending_period_count = sum(1 for t in texts if t.endswith(".") and not t.endswith(".."))
        ending_period_ratio = round(ending_period_count / total, 3)

        # Contractions with apostrophe (e.g. don't vs dont)
        apostrophe_contractions = sum(len(re.findall(r"\b\w+'[a-z]{1,2}\b", t, re.IGNORECASE)) for t in texts)
        slang_contractions = sum(len(re.findall(r"\b(dont|cant|im|didnt|wont|idk|tbh)\b", t, re.IGNORECASE)) for t in texts)
        total_contractions = apostrophe_contractions + slang_contractions
        contraction_apostrophe_ratio = (
            round(apostrophe_contractions / total_contractions, 3) if total_contractions > 0 else 0.0
        )

        # 2. Length & Rhythm
        char_lengths = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        avg_length_chars = round(statistics.mean(char_lengths), 1)
        median_length_chars = round(statistics.median(char_lengths), 1)
        avg_words = round(statistics.mean(word_counts), 1)

        # Bursts (messages sent within 30s of previous message)
        burst_count = 0
        for i in range(1, len(messages)):
            prev_ts = messages[i - 1].get("timestamp_ms", 0)
            curr_ts = messages[i].get("timestamp_ms", 0)
            if curr_ts - prev_ts <= 30000:
                burst_count += 1
        burst_ratio = round(burst_count / max(1, len(messages) - 1), 3)

        # 3. Emojis
        all_emojis: list[str] = []
        messages_with_emojis = 0
        for t in texts:
            found = EMOJI_PATTERN.findall(t)
            if found:
                messages_with_emojis += 1
                for chunk in found:
                    for char in chunk:
                        all_emojis.append(char)

        emoji_density = round(messages_with_emojis / total, 3)
        top_emojis = Counter(all_emojis).most_common(5)

        # 4. Slang & Hinglish detection
        slang_counts: dict[str, int] = Counter()
        hinglish_message_count = 0

        for t in texts:
            lower = t.lower()
            has_hinglish = False
            for slang in SLANG_LEXICON:
                pattern = rf"\b{re.escape(slang)}\b"
                if re.search(pattern, lower):
                    slang_counts[slang] += 1
                    has_hinglish = True
            if has_hinglish:
                hinglish_message_count += 1

        hinglish_ratio = round(hinglish_message_count / total, 3)
        slang_frequencies = {
            k: round(v / total, 4) for k, v in slang_counts.most_common(12)
        }

        # 5. Questions
        questions = sum(1 for t in texts if "?" in t)
        question_frequency = round(questions / total, 3)

        return {
            "total": total,
            "lowercase_ratio": lowercase_ratio,
            "ending_period_ratio": ending_period_ratio,
            "contraction_apostrophe_ratio": contraction_apostrophe_ratio,
            "avg_length_chars": avg_length_chars,
            "median_length_chars": median_length_chars,
            "avg_words": avg_words,
            "burst_ratio": burst_ratio,
            "emoji_density": emoji_density,
            "top_emojis": top_emojis,
            "hinglish_ratio": hinglish_ratio,
            "slang_frequencies": slang_frequencies,
            "question_frequency": question_frequency,
        }

    def analyze(self, operator_messages: list[dict[str, Any]], version_id: str = "v001") -> OperatorStyleProfile:
        """Run complete time-aware style analysis over operator messages."""
        if not operator_messages:
            return OperatorStyleProfile(version_id=version_id, updated_at=datetime.now(timezone.utc).isoformat())

        # Sort chronologically
        sorted_msgs = sorted(operator_messages, key=lambda m: m.get("timestamp_ms", 0))

        # Split into long-term and recent based on timestamp
        now_ms = int(time.time() * 1000)
        recent_threshold_ms = now_ms - (self.recent_window_days * 86400 * 1000)

        recent_msgs = [m for m in sorted_msgs if m.get("timestamp_ms", 0) >= recent_threshold_ms]
        # If dataset is historical and no messages are within 90 days of wall clock, use last 25% as recent
        if not recent_msgs and len(sorted_msgs) >= 10:
            split_idx = int(len(sorted_msgs) * 0.75)
            recent_msgs = sorted_msgs[split_idx:]

        long_term_stats = self.analyze_subset(sorted_msgs)
        recent_stats = self.analyze_subset(recent_msgs) if recent_msgs else long_term_stats

        # Weighted blend function
        def blend(lt_val: float, rec_val: float) -> float:
            return round((lt_val * self.long_term_weight) + (rec_val * self.recent_weight), 3)

        evidence_count = len(sorted_msgs)
        confidence = self.calculate_confidence(evidence_count)

        # Openers analysis (first message in each conversation)
        openers: list[str] = []
        seen_convs: set[str] = set()
        for m in sorted_msgs:
            cid = m.get("conversation_id")
            if cid not in seen_convs:
                seen_convs.add(cid)
                t = m.get("text", "").strip()
                if t and len(t) < 40:
                    openers.append(t.lower())
        common_openers = [op for op, _ in Counter(openers).most_common(5)]

        return OperatorStyleProfile(
            version_id=version_id,
            total_messages=evidence_count,
            hinglish_ratio=blend(long_term_stats.get("hinglish_ratio", 0.0), recent_stats.get("hinglish_ratio", 0.0)),
            english_ratio=round(1.0 - long_term_stats.get("hinglish_ratio", 0.0), 3),
            hindi_ratio=round(long_term_stats.get("hinglish_ratio", 0.0) * 0.4, 3),
            lowercase_ratio=blend(long_term_stats.get("lowercase_ratio", 0.0), recent_stats.get("lowercase_ratio", 0.0)),
            ending_period_ratio=blend(long_term_stats.get("ending_period_ratio", 0.0), recent_stats.get("ending_period_ratio", 0.0)),
            contraction_apostrophe_ratio=long_term_stats.get("contraction_apostrophe_ratio", 0.0),
            avg_message_length_chars=blend(long_term_stats.get("avg_length_chars", 0.0), recent_stats.get("avg_length_chars", 0.0)),
            median_message_length_chars=long_term_stats.get("median_length_chars", 0.0),
            avg_words_per_message=blend(long_term_stats.get("avg_words", 0.0), recent_stats.get("avg_words", 0.0)),
            burst_message_ratio=blend(long_term_stats.get("burst_ratio", 0.0), recent_stats.get("burst_ratio", 0.0)),
            emoji_density=blend(long_term_stats.get("emoji_density", 0.0), recent_stats.get("emoji_density", 0.0)),
            favorite_emojis=recent_stats.get("top_emojis") or long_term_stats.get("top_emojis", []),
            slang_frequencies=long_term_stats.get("slang_frequencies", {}),
            question_frequency=blend(long_term_stats.get("question_frequency", 0.0), recent_stats.get("question_frequency", 0.0)),
            common_openers=common_openers,
            recent_style=recent_stats,
            long_term_style=long_term_stats,
            confidence=confidence,
            evidence_count=evidence_count,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
