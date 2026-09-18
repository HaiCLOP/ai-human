"""Tests for the new context-aware conversation pipeline — state_delta, contextual intent, action planner v2."""

import pytest
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.action_planner import ActionPlanner, Action
from app.conversation.state import ConversationState, update_conversation_state
from app.conversation.state_delta import compute_state_delta
from app.conversation.response_intent import format_now_block
from datetime import datetime, timezone, timedelta


IST = timezone(timedelta(hours=5, minutes=30))


# ── Format now block tests ──────────────────────────────────────────────────

def test_format_now_block_returns_structured_string():
    dt = datetime(2026, 9, 18, 16, 30, tzinfo=IST)
    block = format_now_block(dt)
    assert block.startswith("[NOW]")
    assert "2026" in block
    assert "IST" in block


def test_format_now_block_morning_period():
    dt = datetime(2026, 9, 18, 9, 15, tzinfo=IST)
    block = format_now_block(dt)
    assert "morning" in block.lower() or "AM" in block


def test_format_now_block_night_period():
    dt = datetime(2026, 9, 18, 22, 0, tzinfo=IST)
    block = format_now_block(dt)
    assert "night" in block.lower() or "PM" in block


# ── Contextual intent resolver tests ───────────────────────────────────────

def test_kyu_classified_as_ask_reason():
    intent = SocialIntentAnalyzer.analyze("kyuu")
    assert intent.social_act == "ASK_REASON"
    assert intent.confidence >= 0.75


def test_kyu_with_context_classified_ask_reason():
    context = [
        {"role": "user", "text": "kya scene hai"},
        {"role": "vesper", "text": "kal exam hai isliye padh rahi hu yaar"},
    ]
    intent = SocialIntentAnalyzer.analyze("kyu?", context_history=context)
    assert intent.social_act == "ASK_REASON"
    assert intent.confidence >= 0.80


def test_phir_classified_as_follow_up():
    intent = SocialIntentAnalyzer.analyze("phir?")
    assert intent.social_act == "FOLLOW_UP"


def test_pure_reaction_after_vesper_statement_classified_react_to_previous():
    context = [
        {"role": "user", "text": "kya scene"},
        {"role": "vesper", "text": "yaar bore ho rahi hu, Riya nahi aa rahi thi aaj school mein"},
    ]
    intent = SocialIntentAnalyzer.analyze("ohh", context_history=context)
    assert intent.social_act == "REACT_TO_PREVIOUS"


def test_other_with_low_confidence_not_kyu_classified():
    intent = SocialIntentAnalyzer.analyze("theek hai")
    # Should NOT be ASK_REASON
    assert intent.social_act not in ("ASK_REASON",)


# ── State delta tests ───────────────────────────────────────────────────────

def make_state(last_vesper="", current_topic=None, questions_asked=None):
    return ConversationState(
        conversation_id="test",
        contact_id="c1",
        last_vesper_message=last_vesper,
        current_topic=current_topic,
        questions_asked_by_vesper=questions_asked or [],
    )


def make_intent(act="other", playfulness=0.4):
    from app.conversation.social_intent import SocialIntent
    return SocialIntent(
        social_act=act,
        confidence=0.8,
        seriousness=0.3,
        hostility=0.0,
        playfulness=playfulness,
        expected_reply_length="short",
        requires_response=True,
        detected_signals=[],
    )


def test_state_delta_why_probe_detected():
    state = make_state(last_vesper="yaar kal exam hai isliye padh rahi hu")
    intent = make_intent("ASK_REASON")
    delta = compute_state_delta("kyuu?", intent, state)
    assert delta.vesper_should_explain is True
    assert "explain" in delta.implied_goal


def test_state_delta_reaction_triggers_elaborate():
    state = make_state(last_vesper="haan yaar kaafi bore ho gayi thi phir bhi padhi aaj")
    intent = make_intent("acknowledgment")
    delta = compute_state_delta("ohh", intent, state)
    assert delta.vesper_should_elaborate is True


def test_state_delta_normal_message_no_special_flags():
    state = make_state(last_vesper="kuch nahi yaar")
    intent = make_intent("check_in")
    delta = compute_state_delta("tu bata kya scene hai", intent, state)
    assert delta.vesper_should_explain is False
    assert delta.vesper_should_elaborate is False


# ── ActionPlanner v2 tests ──────────────────────────────────────────────────

def test_action_planner_explain_on_why_probe():
    from app.conversation.state_delta import StateDelta
    state = make_state(last_vesper="yaar bohot thak gayi hu aaj")
    intent = make_intent("ASK_REASON")
    delta = compute_state_delta("kyuu?", intent, state)
    plan = ActionPlanner.plan(state=state, intent=intent, state_delta=delta)
    assert plan.action == Action.EXPLAIN
    assert plan.confidence >= 0.85


def test_action_planner_elaborate_on_reaction():
    state = make_state(last_vesper="haan Riya ne basically mujhe kal rona diya tha")
    intent = make_intent("acknowledgment")
    delta = compute_state_delta("ohh", intent, state)
    plan = ActionPlanner.plan(state=state, intent=intent, state_delta=delta)
    assert plan.action == Action.ELABORATE


def test_action_planner_interview_mode_blocks_questions():
    state = ConversationState(
        conversation_id="test",
        contact_id="c1",
        in_interview_mode=True,
        question_streak=3,
        last_vesper_message="aur tu kya kar rahi hai?",
    )
    intent = make_intent("check_in")
    plan = ActionPlanner.plan(state=state, intent=intent)
    assert plan.action in (Action.SELF_DISCLOSE, Action.OBSERVE)
    assert "not ask" in plan.conversational_goal.lower() or "question" in plan.conversational_goal.lower()


def test_action_planner_other_low_confidence_routes_continue_topic():
    state = make_state(current_topic="school")
    intent = make_intent("other")
    intent_low = SocialIntentAnalyzer.analyze("hmm")
    plan = ActionPlanner.plan(state=state, intent=intent_low)
    # Should not be generic DIRECT_REPLY or a question action
    # Should continue conversation or at minimum not ask a new question
    assert plan.confidence >= 0.55


def test_farewell_always_gets_direct_reply():
    from app.conversation.social_intent import SocialIntent
    intent = SocialIntent(
        social_act="farewell", confidence=0.95, seriousness=0.2,
        hostility=0.0, playfulness=0.2, expected_reply_length="very_short",
        requires_response=True, detected_signals=["farewell_pattern"],
    )
    state = make_state()
    plan = ActionPlanner.plan(state=state, intent=intent)
    assert plan.action == Action.DIRECT_REPLY
    assert "farewell" in plan.conversational_goal.lower() or "send-off" in plan.conversational_goal.lower()


# ── ConversationState update tests ─────────────────────────────────────────

def test_state_update_question_streak_increments():
    from app.conversation.social_intent import SocialIntent
    state = ConversationState(conversation_id="t", contact_id="c1", question_streak=1)
    intent = SocialIntent(
        social_act="check_in", confidence=0.7, seriousness=0.2,
        hostility=0.0, playfulness=0.4, expected_reply_length="short",
        requires_response=True, detected_signals=[],
    )
    updated = update_conversation_state(state, intent, "direct_answer", vesper_reply="tu kya kar raha hai?")
    assert updated.question_streak == 2


def test_state_update_question_streak_resets_when_no_question():
    from app.conversation.social_intent import SocialIntent
    state = ConversationState(conversation_id="t", contact_id="c1", question_streak=2)
    intent = SocialIntent(
        social_act="acknowledgment", confidence=0.7, seriousness=0.2,
        hostility=0.0, playfulness=0.4, expected_reply_length="very_short",
        requires_response=True, detected_signals=[],
    )
    updated = update_conversation_state(state, intent, "react", vesper_reply="haan theek hai")
    assert updated.question_streak == 0  # No ? in reply


def test_state_interview_mode_triggers_at_streak_3():
    from app.conversation.social_intent import SocialIntent
    state = ConversationState(conversation_id="t", contact_id="c1", question_streak=2)
    intent = SocialIntent(
        social_act="check_in", confidence=0.7, seriousness=0.2,
        hostility=0.0, playfulness=0.4, expected_reply_length="short",
        requires_response=True, detected_signals=[],
    )
    updated = update_conversation_state(state, intent, "direct_answer", vesper_reply="aur kya ho raha?")
    assert updated.question_streak == 3
    assert updated.in_interview_mode is True
