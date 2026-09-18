"""User style learning subsystem: stylometric observation, confidence scoring, and adaptation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from app.core.logging import get_logger
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import StyleProfileRecord, StyleRepository

logger = get_logger("memory.style")

# High-frequency informal internet slang markers
SLANG_LEXICON = {
    "ngl", "tbh", "fr", "cap", "lowkey", "highkey", "idk", "rn", "lmao",
    "ded", "bruh", "bro", "smh", "sus", "bet", "nah", "yea", "omg", "wtf",
    "fml", "cooked", "delulu", "flex", "wbu", "hbu"
}

# Common Romanized Hindi/Urdu code-switching markers (Hinglish)
HINGLISH_LEXICON = {
    "yaar", "arre", "bhai", "sahi", "kya", "chal", "matlab", "funda",
    "scene", "batao", "acha", "accha", "achha", "theek", "thik", "mast",
    "boss", "arrey", "haan", "haa", "hnn", "nahi", "nhi", "nai", "dekh",
    "thoda", "pata", "pta", "bol", "jugaad", "load", "bakwaas", "dimag",
    "hadd", "sach", "kasam", "chull", "vibe", "fat", "phat", "pakau", "chhod"
}

# Emoji detection pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class StylometricFeatures:
    word_count: int
    is_all_lowercase: bool
    is_all_caps: bool
    has_trailing_ellipses: bool
    lacks_punctuation: bool
    slang_words: list[str]
    hinglish_words: list[str]
    emoji_count: int


class StyleAnalyzer:
    """Extracts linguistic and stylometric features from a single message."""

    @staticmethod
    def analyze_message(text: str) -> StylometricFeatures:
        clean = text.strip()
        words = re.findall(r"[a-zA-Z0-9']+", clean)
        word_count = len(words)

        is_lowercase = clean.islower() and any(c.isalpha() for c in clean)
        is_caps = clean.isupper() and any(c.isalpha() for c in clean)
        trailing_ellipses = clean.endswith("...") or clean.endswith("..")
        lacks_punc = not clean.endswith((".", "!", "?", "..."))

        words_lower = [w.lower() for w in words]
        slang_found = [w for w in words_lower if w in SLANG_LEXICON]
        hinglish_found = [w for w in words_lower if w in HINGLISH_LEXICON]
        emojis_found = EMOJI_PATTERN.findall(clean)

        return StylometricFeatures(
            word_count=word_count,
            is_all_lowercase=is_lowercase,
            is_all_caps=is_caps,
            has_trailing_ellipses=trailing_ellipses,
            lacks_punctuation=lacks_punc,
            slang_words=slang_found,
            hinglish_words=hinglish_found,
            emoji_count=len(emojis_found),
        )


class StyleLearner:
    """Accumulates observations over multiple turns and produces prompt adaptation guidelines."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.repo = StyleRepository(self.db)
        self.analyzer = StyleAnalyzer()

    def observe_message(self, conversation_id: str, message_text: str) -> None:
        """Process inbound user message and update stylometric profile."""
        profile = self.repo.get_or_create_profile(conversation_id)
        features = self.analyzer.analyze_message(message_text)

        if features.word_count == 0:
            return

        # 1. Update sentence length: directly initialize on first turn, otherwise use EMA (alpha = 0.25)
        is_first_observation = (profile.slang_vocabulary_json == "{}" and profile.avg_sentence_length == 8.0)
        if is_first_observation:
            new_avg_len = float(features.word_count)
        else:
            new_avg_len = round(0.25 * features.word_count + 0.75 * profile.avg_sentence_length, 2)

        # 2. Update Emoji density (emojis per word)
        current_emoji_density = features.emoji_count / max(1, features.word_count)
        new_emoji_density = round(0.25 * current_emoji_density + 0.75 * profile.emoji_density, 3)

        # 3. Slang vocabulary frequency
        try:
            slang_dict: dict[str, int] = json.loads(profile.slang_vocabulary_json)
        except Exception:
            slang_dict = {}

        for sw in features.slang_words:
            slang_dict[sw] = slang_dict.get(sw, 0) + 1
            # Record observation with confidence scaled by evidence count
            count = slang_dict[sw]
            conf = min(1.0, count / 4.0)
            self.repo.record_observation(profile.profile_id, "slang", sw, confidence=conf)

        for hw in features.hinglish_words:
            slang_dict[hw] = slang_dict.get(hw, 0) + 1
            count = slang_dict[hw]
            conf = min(1.0, count / 3.0)
            self.repo.record_observation(profile.profile_id, "hinglish", hw, confidence=conf)

        if features.is_all_lowercase:
            self.repo.record_observation(profile.profile_id, "casing", "all_lowercase", confidence=0.7)

        # Formality heuristic: high sentence length + low slang = higher formality
        formality = 0.2
        if new_avg_len > 20 and not features.slang_words:
            formality = 0.6
        elif new_avg_len <= 5 and (features.is_all_lowercase or features.slang_words):
            formality = 0.1

        # Commit profile update
        self.repo.update_profile(
            profile_id=profile.profile_id,
            avg_sentence_length=new_avg_len,
            formality_score=formality,
            emoji_density=new_emoji_density,
            slang_vocabulary_json=json.dumps(slang_dict),
        )

        logger.info(
            "style.observed",
            conversation_id=conversation_id,
            avg_length=new_avg_len,
            formality=formality,
            slang_count=len(features.slang_words),
            hinglish_count=len(features.hinglish_words),
        )

    def get_style_notes_for_prompt(self, conversation_id: str) -> list[str]:
        """Compile high-confidence stylometric traits into concise prompt bullet points."""
        profile = self.repo.get_or_create_profile(conversation_id)
        observations = self.repo.get_observations(profile.profile_id)

        notes: list[str] = []

        # Sentence length guidance
        if profile.avg_sentence_length <= 6:
            notes.append("User writes very terse messages (1-5 words). Keep reply brief and punchy.")
        elif profile.avg_sentence_length > 20:
            notes.append("User writes longer messages. Feel free to use 2-3 full sentences.")

        # Slang guidance (only report high-confidence traits >= 0.5)
        high_conf_slang = [obs.trait_value for obs in observations if obs.trait_name == "slang" and obs.confidence >= 0.5]
        if high_conf_slang:
            notes.append(f"User frequently uses informal slang ({', '.join(high_conf_slang[:4])}). Match casual posture naturally.")

        # Hinglish guidance
        high_conf_hinglish = [obs.trait_value for obs in observations if obs.trait_name == "hinglish" and obs.confidence >= 0.5]
        if high_conf_hinglish:
            notes.append(f"User incorporates conversational Hinglish/code-switching ({', '.join(high_conf_hinglish[:3])}).")

        # Casing / Punctuation
        casing_obs = [obs for obs in observations if obs.trait_name == "casing" and obs.trait_value == "all_lowercase" and obs.confidence >= 0.5]
        if casing_obs:
            notes.append("User writes almost entirely in lowercase without standard capitalization.")

        if not notes:
            notes.append("User style is neutral/conversational. Maintain default deadpan cadence.")

        return notes
