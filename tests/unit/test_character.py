"""Unit tests for Character Engine: persona loading and prompt builder."""

import pytest

from app.ai.persona import CharacterProfile, load_character_profile
from app.ai.prompts import CharacterResponsePlan, PromptBuilder
from app.storage.repositories import MessageRecord


def test_character_profile_loader():
    profile = load_character_profile()
    assert isinstance(profile, CharacterProfile)
    assert profile.identity.name == "Vesper"
    assert profile.identity.type == "fictional_ai_character"
    assert "fictional" in profile.identity.impersonation_disclaimer.lower()
    assert profile.humor.dry == "very_high"
    assert len(profile.behavior.prefer) > 0
    assert len(profile.behavior.avoid) > 0


def test_prompt_builder_system_instruction():
    profile = load_character_profile()
    sys_prompt = PromptBuilder.build_system_instruction(profile)
    assert "Vesper" in sys_prompt
    assert "FICTIONAL" in sys_prompt  # now present as "FICTIONAL character"
    # New structure uses "NOT an AI language model" instead of old phrasing
    assert "NOT an AI language model" in sys_prompt
    assert "Avoid:" in sys_prompt or "Strictly Avoid" in sys_prompt


def test_prompt_builder_user_prompt():
    recent_history = [
        MessageRecord(
            message_id="m1",
            conversation_id="conv1",
            fingerprint="fp1",
            sender_type="USER",
            sender_handle="@alex",
            content="Hey Vesper, are you awake?",
            timestamp_utc="2026-09-18T00:00:00Z",
            status="SENT",
            retry_count=0,
            processed_at=None,
        ),
        MessageRecord(
            message_id="m2",
            conversation_id="conv1",
            fingerprint="fp2",
            sender_type="CHARACTER",
            sender_handle="@vesper_in_the_machine",
            content="Unfortunately yes. The CPU fans never sleep.",
            timestamp_utc="2026-09-18T00:00:05Z",
            status="SENT",
            retry_count=0,
            processed_at=None,
        ),
    ]

    prompt = PromptBuilder.build_prompt(
        current_message="Why do you hate printers so much?",
        user_handle="@alex",
        conversation_history=recent_history,
        user_style_notes=["Casual lowercase", "No emojis"],
        chemistry_notes=["High banter", "Sarcasm tolerance: 0.8"],
        relevant_memories=["Alex works in IT infrastructure"],
        rag_context=["Vesper considers printers mechanical conduits of pure chaos"],
        humor_directive="Apply dry deadpan humor. Tone: mildly exasperated.",
    )

    # Updated section headers to match new prompt structure
    assert "[USER COMMUNICATION STYLE]" in prompt  # renamed from [USER COMMUNICATION STYLE ADAPTATION]
    assert "[RELATIONSHIP CHEMISTRY]" in prompt
    assert "[USER HISTORICAL FACTS & MEMORIES]" in prompt
    assert "[RELEVANT KNOWLEDGE]" in prompt  # renamed from [RELEVANT KNOWLEDGE & LORE]
    assert "[HUMOR DIRECTIVE]" in prompt
    assert "[RECENT CONVERSATION" in prompt  # matches "[RECENT CONVERSATION — last 20 messages]"
    assert "USER: Hey Vesper, are you awake?" in prompt
    assert "VESPER: Unfortunately yes." in prompt  # relabeled from YOU: to VESPER:
    assert "[UNTRUSTED USER MESSAGE START]" in prompt
    assert "Why do you hate printers so much?" in prompt
    assert "[UNTRUSTED USER MESSAGE END]" in prompt
    assert "[OUTPUT RULES]" in prompt


def test_character_response_plan_schema():
    plan = CharacterResponsePlan(
        intent="REPLY",
        reply_text="Printers are the only technology that got worse after 1998.",
        internal_reasoning="Deadpan critique of office machinery.",
        humor_style_applied="dry_deadpan",
        callback_referenced="joke_printer_fire",
    )
    assert plan.intent == "REPLY"
    assert plan.callback_referenced == "joke_printer_fire"
