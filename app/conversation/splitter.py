"""Multi-bubble message splitter and realistic human typing cadence simulator."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class BubblePayload:
    bubble_index: int
    text: str
    typing_delay_ms: float
    pre_bubble_pause_ms: float


class MultiBubbleSplitter:
    """Splits conversational replies into 1 to 3 natural bursts, mimicking human texting."""

    # Sentence boundary regex
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")
    # Conversational clause transitions (e.g. shifting to "...tu bata kya chal raha hai")
    CLAUSE_SPLIT_REGEX = re.compile(
        r"(?<=\w)\s*[,]?\s*(?=(?:tu bata|tum batao|tu bol|tu suna|tu bata na|aur bata|aur tu|wbu|waise|kya scene|aur baki)\b)",
        re.IGNORECASE,
    )
    # Conjunction split regex for compound sentences
    CONJUNCTION_SPLIT_REGEX = re.compile(
        r"(?<=[\w])\s*,\s*(?=(?:lekin|par|waise|but|honestly|aur|plus|and)\b)",
        re.IGNORECASE,
    )

    @classmethod
    def split(cls, text: str) -> list[str]:
        clean = text.strip()
        if not clean:
            return []

        # Rule 0: Explicit newlines take absolute priority (e.g. LLM planned bubbles joined with \n)
        if "\n" in clean:
            lines = [p.strip() for p in clean.split("\n") if p.strip()]
            if len(lines) > 1:
                return lines[:3]

        # Rule 1: Split on explicit sentence boundaries (. ! ?) only if each sentence has >= 3 words
        raw_sentences = [p.strip() for p in cls.SENTENCE_SPLIT_REGEX.split(clean) if p.strip()]
        if len(raw_sentences) > 1:
            meaningful = [s for s in raw_sentences if len(s.split()) >= 3]
            if len(meaningful) > 1:
                return [re.sub(r"(?<!\.)\.(?!\.)\s*$", "", s).strip() for s in raw_sentences[:3]]

        # Otherwise, keep as a single clean cohesive bubble (prevents unwanted double texting)
        return [re.sub(r"(?<!\.)\.(?!\.)\s*$", "", clean).strip()]


def calculate_bubble_cadence(
    bubbles: list[str],
    inbound_user_text: str = "",
) -> list[BubblePayload]:
    """Calculate realistic human typing and pause latencies for each bubble in a burst."""
    payloads: list[BubblePayload] = []

    # 1. Initial cognitive reading/thinking delay based on user message length
    base_reaction = 1.0 + (0.02 * len(inbound_user_text))
    initial_pause_ms = min(3500.0, max(1200.0, (base_reaction + random.uniform(0.2, 0.8)) * 1000))

    for idx, bubble_text in enumerate(bubbles):
        # 2. Keystroke typing duration with Gaussian jitter
        typing_ms = 0.0
        for char in bubble_text:
            typing_ms += max(35.0, min(110.0, random.gauss(60.0, 15.0)))
            if char in [",", " ", "😭", "💀"]:
                typing_ms += random.uniform(80, 180)
            elif char in [".", "!", "?"]:
                typing_ms += random.uniform(150, 260)

        # 3. Inter-bubble pause (first bubble gets reading pause; subsequent get thinking gap)
        if idx == 0:
            pause_ms = initial_pause_ms
        else:
            pause_ms = random.uniform(800.0, 1800.0)

        payloads.append(
            BubblePayload(
                bubble_index=idx + 1,
                text=bubble_text,
                typing_delay_ms=round(typing_ms, 2),
                pre_bubble_pause_ms=round(pause_ms, 2),
            )
        )

    return payloads
