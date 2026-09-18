"""Deterministic post-processing humanizer pipeline for authentic Indian youth / Gen Z texting."""

from __future__ import annotations

import re
from typing import Sequence

# Allowed emojis — skull is permanently banned, only these two are ever permitted
ALLOWED_EMOJIS = {"😭", "😂"}

# Comprehensive regex matching emojis across Unicode ranges
EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010ffff"  # SMP emojis
    r"\u2600-\u27bf"           # Misc symbols, dingbats
    r"\ufe0f"                  # Variation selectors
    r"\u200d"                  # Zero-width joiner
    r"]+",
    flags=re.UNICODE,
)

# Rhetorical Q&A monologue patterns ("Kyu? Kyunki...", "Why? Because...")
RHETORICAL_QA_PATTERNS = [
    re.compile(
        r"(?:(?:pata\s+hai\s+)?(?:kyu|kyun)\s*\??\s*)+[,.\s]*(?:kyunki|kyuki|kyoki)\b\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:guess\s+why|you\s+know\s+why|why)\s*\??\s*)+[,.\s]*(?:because|bcoz|coz|cuz)\b\s*",
        re.IGNORECASE,
    ),
]

# Trailing single period at the end of a message
TRAILING_PERIOD_REGEX = re.compile(r"(?<!\.)\.(?!\.)\s*$")

# Period immediately preceding an emoji
PERIOD_BEFORE_EMOJI_REGEX = re.compile(r"(?<!\.)\.(?!\.)(?=\s*[\U00010000-\U0010ffff])")

# Acronyms preserved in uppercase
PRESERVED_ACRONYMS = {
    "CBSE", "JEE", "NEET", "NCERT", "WTF", "LMAO", "LOL", "IDK", "NGL",
    "TBH", "DM", "IG", "AI", "OK", "BKL", "MC", "BC", "PVR", "IIT", "CUET",
    "PW", "FIITJEE", "ALLEN"
}

# Apostrophe contractions to strip apostrophes from
CONTRACTION_PATTERNS = [
    (re.compile(r"\b(can|don|won|didn|isn|aren|haven|couldn|wouldn|shouldn|wasn|weren)'t\b", re.IGNORECASE), r"\1t"),
    (re.compile(r"\b(I|you|they|we)'m\b", re.IGNORECASE), r"\1m"),
    (re.compile(r"\b(it|that|what|there|here|who)'s\b", re.IGNORECASE), r"\1s"),
    (re.compile(r"\b(I|you|they|we|he|she)'ll\b", re.IGNORECASE), r"\1ll"),
    (re.compile(r"\b(I|you|they|we)'ve\b", re.IGNORECASE), r"\1ve"),
    (re.compile(r"\b(I|you|they|we|he|she)'d\b", re.IGNORECASE), r"\1d"),
]


def clean_rhetorical_qa(text: str) -> str:
    """Detect and rewrite theatrical 'Kyu? Kyunki...' or 'Why? Because...' patterns."""
    cleaned = text
    for pattern in RHETORICAL_QA_PATTERNS:
        matches = list(pattern.finditer(cleaned))
        if not matches:
            continue

        for m in reversed(matches):
            start, end = m.span()
            prefix = cleaned[:start].strip()
            suffix = cleaned[end:].strip()

            if prefix:
                prefix_clean = prefix.rstrip(".!?")
                cleaned = f"{prefix_clean} kyunki {suffix}"
            else:
                cleaned = f"bas {suffix}"

    return cleaned


def filter_emojis(text: str, allow_emoji: bool = True) -> str:
    """Filter emojis: strip all forbidden emojis, limit to max 1 allowed emoji at the end.
    
    Skull emoji (💀) is ALWAYS stripped regardless of any setting.
    """
    if not text:
        return ""
    if not allow_emoji:
        return re.sub(r"[^\S\r\n]+", " ", EMOJI_REGEX.sub("", text)).strip()

    first_allowed_emoji: str | None = None

    def emoji_evaluator(match: re.Match[str]) -> str:
        nonlocal first_allowed_emoji
        matched_str = match.group(0)

        for char in matched_str:
            if char in ALLOWED_EMOJIS and first_allowed_emoji is None:
                first_allowed_emoji = char
                return f" {char} "
        return ""  # strip all other emojis (including skull)

    sanitized = EMOJI_REGEX.sub(emoji_evaluator, text)
    sanitized = re.sub(r"[^\S\r\n]+", " ", sanitized).strip()
    return sanitized


def normalize_lowercase(text: str, preserve_acronyms: bool = True) -> str:
    """Normalize text to lowercase while selectively preserving critical acronyms."""
    if not text:
        return ""

    if not preserve_acronyms:
        return text.lower()

    tokens = re.split(r"(\s+|[.,!?;:'\"()]+)", text)
    processed: list[str] = []

    for token in tokens:
        if token.upper() in PRESERVED_ACRONYMS:
            processed.append(token.upper())
        else:
            processed.append(token.lower())

    return "".join(processed)


def strip_ending_punctuation(text: str) -> str:
    """Strip trailing periods from casual DMs while strictly preserving ellipses and interrogatives."""
    if not text:
        return ""
    # Strip period preceding a trailing emoji
    text = PERIOD_BEFORE_EMOJI_REGEX.sub("", text)
    # Strip single trailing period at end of string
    text = TRAILING_PERIOD_REGEX.sub("", text)
    return text.strip()


def strip_apostrophes(text: str) -> str:
    """Strip apostrophes from standard contractions (e.g. don't -> dont, I'm -> im)."""
    if not text:
        return ""
    result = text
    for pattern, replacement in CONTRACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def humanize_text(
    text: str,
    lowercase: bool = True,
    allow_skull: bool = False,  # kept for backwards compat — skull is ALWAYS stripped
    allow_emoji: bool = True,
) -> str:
    """Orchestrate full deterministic humanization pipeline on a candidate reply."""
    if not text:
        return ""

    # 1. Clean rhetorical Q&A monologues
    step1 = clean_rhetorical_qa(text)

    # 2. Filter emojis (skull always banned, only 😭 / 😂 ever allowed)
    step2 = filter_emojis(step1, allow_emoji=allow_emoji)

    # 3. Strip apostrophes in contractions
    step3 = strip_apostrophes(step2)

    # 4. Lowercase normalizer
    step4 = normalize_lowercase(step3) if lowercase else step3

    # 5. Strip trailing periods
    step5 = strip_ending_punctuation(step4)

    return step5
