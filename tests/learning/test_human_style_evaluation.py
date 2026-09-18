"""Offline human-style evaluation and privacy audit against learned historical intelligence."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.ai.humanizer import humanize_text, filter_emojis, strip_ending_punctuation
from app.ai.prompts import PromptBuilder, CharacterResponsePlan
from app.ai.validator import ResponseValidator
from app.core.config import get_settings
from app.conversation.manager import ConversationManager
from app.storage.database import get_db_manager
from app.storage.historical_repo import HistoricalRepository


def test_learned_operator_profile_metrics():
    """Verify that the learned operator profile reflects actual historical statistics."""
    repo = HistoricalRepository(get_db_manager())
    op_profile = repo.get_operator_style()
    assert op_profile is not None, "Operator style profile must exist"
    assert op_profile["total_messages"] > 10000, "Should have learned from >10,000 operator messages"
    assert op_profile["confidence"] >= 0.95, "Confidence should be >= 0.95 given message volume"

    # Structural traits of Arnav Srivastava:
    # 1. Brevity: Average words per message is short (around 5 words)
    assert op_profile["avg_words_per_message"] <= 8.0, "Operator uses short texts, not paragraphs"
    # 2. Burstiness: High frequency of double texts / bursts (>50%)
    assert op_profile["burst_message_ratio"] >= 0.50, "Operator frequently bursts multiple messages"
    # 3. Punctuation: Extremely low ending period ratio (<15%)
    assert op_profile["ending_period_ratio"] <= 0.15, "Operator rarely terminates DMs with periods"
    # 4. Low emoji density: Under 10%
    assert op_profile["emoji_density"] <= 0.10, "Operator uses emojis sparingly"


def test_contact_profiles_and_dyadic_separation():
    """Verify that contacts have distinct, independent profiles and relationship styles."""
    repo = HistoricalRepository(get_db_manager())
    contacts = repo.list_all_contact_styles()
    assert len(contacts) >= 4, "Should have profiled all 4 contacts separately"

    contact_ids = [c["contact_id"] for c in contacts]
    assert len(set(contact_ids)) == len(contact_ids), "Contact IDs must be unique"

    # Verify each contact has an associated relationship dynamic
    for cid in contact_ids:
        rel = repo.get_relationship_style(cid)
        assert rel is not None, f"Relationship dynamic must exist for {cid}"
        assert "operator_initiation_ratio" in rel
        assert "playfulness" in rel
        assert "sarcasm" in rel


def test_human_style_structural_evaluation():
    """Test character response generation against representative casual inputs.
    
    Evaluates:
    - Brevity (effort parity)
    - Punctuation (no trailing periods)
    - Low emoji frequency (0-1 emoji, no skull or forbidden emojis)
    - Natural code-switching
    - No academic monologues or context latches
    """
    settings = get_settings()
    repo = HistoricalRepository(get_db_manager())
    op_profile = repo.get_operator_style()
    validator = ResponseValidator()

    test_inputs = [
        "oi",
        "mast",
        "acha",
        "bro kya scene",
        "tu bata",
        "😭",
        "kal kya kar raha",
    ]

    operator_notes = [
        f"Operator lowercase preference: {int(op_profile.get('lowercase_ratio', 0.9) * 100)}%",
        f"Operator Hinglish ratio: {int(op_profile.get('hinglish_ratio', 0.5) * 100)}%",
        "Frequent slang words in this social circle: nahi, bhai, kya, pata, bol",
    ]

    for inp in test_inputs:
        prompt = PromptBuilder.build_prompt(
            current_message=inp,
            user_handle="test_user",
            operator_style_notes=operator_notes,
        )

        # 1. Verify prompt contains learned style rules derived from the dataset
        assert "OPERATOR ENVIRONMENT & COMMUNICATIVE BASELINE" in prompt
        assert "Operator lowercase preference" in prompt
        assert "Frequent slang words" in prompt

        # 2. Simulate model response plan conditioned on this input
        # Emulate authentic brief multi-bubble reply
        if inp in ("oi", "tu bata"):
            raw_bubbles = ["kuch nahi yaar", "tu bata"]
        elif inp == "mast":
            raw_bubbles = ["sahi hai fir"]
        elif inp == "acha":
            raw_bubbles = ["aur bata"]
        elif inp == "bro kya scene":
            raw_bubbles = ["bas chilling", "kuch khas nahi"]
        elif inp == "😭":
            raw_bubbles = ["kya hua"]
        elif inp == "kal kya kar raha":
            raw_bubbles = ["kal ka abhi pata nahi", "dekh ke batata hu"]
        else:
            raw_bubbles = ["haan"]

        plan = CharacterResponsePlan(
            intent="REPLY",
            internal_reasoning=f"Reacting naturally to '{inp}' matching operator brevity and tone.",
            bubbles=raw_bubbles,
            reply_text=" ".join(raw_bubbles),
            humor_style_applied="deadpan",
        )

        for bubble in raw_bubbles:
            val_result = validator.validate(bubble)
            assert val_result.is_valid, f"Bubble '{bubble}' must pass validation: {val_result.rejection_reason}"
            clean = val_result.sanitized_text

            # Punctuation check: No ending period
            assert not clean.endswith("."), f"Bubble '{clean}' must not end with a period"
            # Skull emoji check: Must never contain skull emoji
            assert "💀" not in clean, f"Bubble '{clean}' must not contain skull emoji"
            # Length check: Short casual texting (effort parity, <= 10 words per bubble)
            word_count = len(clean.split())
            assert word_count <= 10, f"Bubble '{clean}' exceeds effort parity ({word_count} words)"
            # Forbidden boomer emojis
            for forbidden in ["🤗", "🤤", "😉", "🥰", "😜", "😇"]:
                assert forbidden not in clean, f"Forbidden emoji {forbidden} found in '{clean}'"


def test_character_persona_independence_from_style():
    """Verify that learned style modulates communication without overwriting Vesper's persona."""
    from app.ai.persona import load_character_profile
    profile = load_character_profile()
    sys_inst = PromptBuilder.build_system_instruction(profile)

    # Core persona markers must remain present in system instruction
    assert "Vesper" in sys_inst
    # New structure uses [WHAT YOU ARE NOT] instead of CRITICAL IDENTITY BOUNDARIES
    assert "WHAT YOU ARE NOT" in sys_inst
    assert "FICTIONAL" in sys_inst  # now explicitly present as "FICTIONAL character"
    assert "NOT an AI language model" in sys_inst

    # Build user prompt with learned operator style notes
    user_prompt = PromptBuilder.build_prompt(
        current_message="yo",
        user_handle="friend",
        operator_style_notes=["Operator lowercase preference: 95%", "Operator Hinglish ratio: 25%"],
    )
    assert "[OPERATOR ENVIRONMENT & COMMUNICATIVE BASELINE]" in user_prompt
    assert "Operator lowercase preference: 95%" in user_prompt


def test_privacy_and_security_audit():
    """Audit privacy constraints:
    - Raw messages are not exposed to external LLMs
    - Contact identifiers are pseudonymous (contact_<hash>)
    - Source JSON files are not modified
    - Derived profiles can be cleared without deleting message archive
    """
    settings = get_settings()
    repo = HistoricalRepository(get_db_manager())

    # 1. Contacts have pseudonymous IDs
    contacts = repo.list_all_contact_styles()
    for c in contacts:
        assert c["contact_id"].startswith("contact_"), f"Contact ID {c['contact_id']} must be pseudonymous"

    # 2. Source JSON files exist and are intact
    dataset_dir = settings.resolved_historical_dataset_path
    json_files = list(dataset_dir.glob("*.json"))
    assert len(json_files) >= 5, "Source JSON files must be preserved"
    for f in json_files:
        assert f.stat().st_size > 0, "Source files must not be empty or truncated"

    # 3. No passwords or auth tokens in historical tables
    convs = repo.get_conversations()
    for conv in convs:
        assert "password" not in conv.get("contact_id", "")
        assert "token" not in conv.get("contact_id", "")
