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
    # New structure: "NOT an AI" (compact form) and "fictional person"
    assert "NOT an AI" in sys_prompt
    assert "fictional person" in sys_prompt
    assert "Strictly Avoid" in sys_prompt


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
            sender_handle="@vesperdelhi",
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
        relevant_memories=["Alex works in IT infrastructure"],
        rag_context=["Vesper considers printers mechanical conduits of pure chaos"],
    )

    # Context-first: conversation comes first
    assert "[CURRENT CONVERSATION]" in prompt
    assert "USER: Hey Vesper, are you awake?" in prompt
    assert "VESPER: Unfortunately yes." in prompt
    # Current message block
    assert "[CURRENT MESSAGE]" in prompt
    assert "Why do you hate printers so much?" in prompt
    # Established facts (memories)
    assert "[ESTABLISHED FACTS]" in prompt
    # Output rules always present
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
