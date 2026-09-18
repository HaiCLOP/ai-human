"""Unit tests for the Indian Gen Z Text Humanizer and Multi-Bubble Splitter."""

from __future__ import annotations

import pytest

from app.ai.humanizer import (
    clean_rhetorical_qa,
    filter_emojis,
    humanize_text,
    normalize_lowercase,
    strip_apostrophes,
    strip_ending_punctuation,
)
from app.conversation.splitter import MultiBubbleSplitter, calculate_bubble_cadence
from app.memory.repetition import TopicSaturationCache


def test_clean_rhetorical_qa():
    # Compound rhetorical monologue rewrite
    raw1 = "kal nahi aa paungi. Kyu? Kyunki sharma sir ne extra class rakh li"
    cleaned1 = clean_rhetorical_qa(raw1)
    assert "kyu? kyunki" not in cleaned1.lower()
    assert "kyunki sharma sir" in cleaned1.lower()

    # Standalone rhetorical opener
    raw2 = "Kyun? Kyunki kal deadline hai"
    cleaned2 = clean_rhetorical_qa(raw2)
    assert cleaned2.lower().startswith("bas kal deadline")


def test_filter_emojis_strips_cringe_and_limits_to_one():
    # Forbidden emojis must be stripped
    text1 = "Arre in quadratic equations ka romance hai 🤗🤤"
    assert filter_emojis(text1) == "Arre in quadratic equations ka romance hai"

    # Multiple allowed emojis must be collapsed to max 1
    text2 = "bhai im dead 😭😭😭"
    assert filter_emojis(text2) == "bhai im dead 😭"

    # Skull is now ALWAYS banned — stripped entirely; 😂 (allowed) comes through instead
    text3 = "dead 💀😂"
    assert filter_emojis(text3) == "dead 😂"

    # Unallowed emojis stripped while preserving allowed
    text4 = "kya baat hai 😊 😭 ✨"
    assert filter_emojis(text4) == "kya baat hai 😭"


def test_normalize_lowercase_preserves_acronyms():
    raw = "Bhai CBSE and JEE are giving me an existential crisis"
    norm = normalize_lowercase(raw, preserve_acronyms=True)
    assert norm.startswith("bhai CBSE and JEE")
    assert "crisis" in norm


def test_strip_ending_punctuation():
    # Single trailing period stripped
    assert strip_ending_punctuation("kuch nahi yaar.") == "kuch nahi yaar"
    assert strip_ending_punctuation("dead hu. 😭") == "dead hu 😭"

    # Ellipses and questions preserved
    assert strip_ending_punctuation("wait what...") == "wait what..."
    assert strip_ending_punctuation("kya scene hai?") == "kya scene hai?"
    assert strip_ending_punctuation("kya scene hai??") == "kya scene hai??"


def test_strip_apostrophes():
    assert strip_apostrophes("I don't think I'm ready, can't even sleep") == "I dont think Im ready, cant even sleep"


def test_humanize_text_end_to_end():
    # Simulate the exact failure from the user's screenshot
    ai_raw = "Mast toh tab hoga jab ye equations solve ho jayenge, yaar. Abhi bhi wrestle kar rahi hu. Kyu? Kyunki kal deadline hai. 🤗"
    humanized = humanize_text(ai_raw)

    # 1. No hugging face emoji
    assert "🤗" not in humanized
    # 2. No trailing period
    assert not humanized.endswith(".")
    # 3. No "kyu? kyunki"
    assert "kyu? kyunki" not in humanized.lower()
    # 4. Lowercase
    assert humanized.islower()


def test_multi_bubble_splitter():
    # Short message stays single bubble
    short = "kuch nahi yaar"
    assert MultiBubbleSplitter.split(short) == ["kuch nahi yaar"]

    # Compound multi-sentence message splits into 2-3 bubbles
    compound = "kuch nahi yaar. bas assignments khatam kar rahi hu 😭. tu bata"
    bubbles = MultiBubbleSplitter.split(compound)
    assert len(bubbles) == 3
    assert bubbles[0] == "kuch nahi yaar"
    assert "assignments" in bubbles[1]
    assert bubbles[2] == "tu bata"

    # Never exceed 3 bubbles
    long_text = "bubble one here. bubble two here. bubble three here. bubble four here"
    split_bubbles = MultiBubbleSplitter.split(long_text)
    assert len(split_bubbles) <= 3

    # Unpunctuated continuous sentences stay as a single cohesive bubble (prevents unwanted double texting)
    sentence = "theek hai na abhi bas normal hi rahe tu bata kya chal raha hai"
    split_res = MultiBubbleSplitter.split(sentence)
    assert split_res == ["theek hai na abhi bas normal hi rahe tu bata kya chal raha hai"]

    # Explicit newlines from planned bubbles split cleanly
    newline_res = MultiBubbleSplitter.split("bubble one here\nbubble two here")
    assert newline_res == ["bubble one here", "bubble two here"]


def test_calculate_bubble_cadence():
    bubbles = ["kuch nahi yaar", "bas assignments chal rahe 😭", "tu bata"]
    cadence = calculate_bubble_cadence(bubbles, inbound_user_text="Mast")
    assert len(cadence) == 3
    assert cadence[0].bubble_index == 1
    assert cadence[0].pre_bubble_pause_ms >= 1200.0
    assert cadence[1].pre_bubble_pause_ms >= 800.0


def test_topic_saturation_cache():
    cache = TopicSaturationCache()
    # Mention quadratic equations repeatedly
    cache.register_character_message("arre yaar quadratic equations solve kar rahi hu")
    cache.register_character_message("ab tak equations hi chal rahi hai")

    saturated = cache.get_saturated_topics(threshold=2)
    assert "quadratic equations" in saturated

    # Strip repetitive opener
    stripped = cache.strip_repetitive_opener("arre yaar kuch naya batao")
    assert not stripped.startswith("arre yaar")
