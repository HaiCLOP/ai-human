"""Unit tests for topic decay, closer handling, and temporal action planning."""

import pytest
from app.conversation.state import ConversationState, update_conversation_state
from app.conversation.social_intent import SocialIntentAnalyzer
from app.conversation.action_planner import ActionPlanner, Action
from app.conversation.state_delta import compute_state_delta


def test_topic_decay_after_two_turns_without_mention():
    state = ConversationState(conversation_id="test_decay")

    # Turn 1: Introduce entertainment topic
    intent1 = SocialIntentAnalyzer.analyze("Movie dekh raha hu")
    state = update_conversation_state(
        state=state,
        intent=intent1,
        strategy="react",
        vesper_reply="kaunsi movie?",
        current_user_text="Movie dekh raha hu",
    )
    assert state.current_topic == "entertainment"
    assert state.topic_turn_age == 0

    # Turn 2: Non-topic message (e.g. acknowledgment)
    intent2 = SocialIntentAnalyzer.analyze("theek hai")
    state = update_conversation_state(
        state=state,
        intent=intent2,
        strategy="react",
        vesper_reply="haan",
        current_user_text="theek hai",
    )
    assert state.topic_turn_age == 1
    assert state.current_topic == "entertainment"

    # Turn 3: Another non-topic message -> topic must expire to None!
    intent3 = SocialIntentAnalyzer.analyze("sahi hai")
    state = update_conversation_state(
        state=state,
        intent=intent3,
        strategy="react",
        vesper_reply="haan",
        current_user_text="sahi hai",
    )
    assert state.topic_turn_age >= 2
    assert state.current_topic is None
    assert len(state.topic_stack) == 0


def test_action_planner_does_not_force_self_disclose_on_okayyy_in_interview_mode():
    state = ConversationState(
        conversation_id="test_interview",
        in_interview_mode=True,
        question_streak=3,
    )
    intent = SocialIntentAnalyzer.analyze("Okayyy")
    assert intent.social_act == "acknowledgment"

    delta = compute_state_delta("Okayyy", intent, state)
    plan = ActionPlanner.plan(state=state, intent=intent, relationship_profile=None, mood=None, state_delta=delta)

    # Must NOT be SELF_DISCLOSE
    assert plan.action == Action.REACT
    assert plan.allow_multi_bubble is False
    assert plan.brevity_target == "very_short"
    assert "1-word closer" in plan.conversational_goal or "haan" in plan.conversational_goal


def test_action_planner_answers_time_query_cleanly():
    state = ConversationState(conversation_id="test_time")
    intent = SocialIntentAnalyzer.analyze("What's the time")
    assert intent.social_act == "question_logistical"

    delta = compute_state_delta("What's the time", intent, state)
    plan = ActionPlanner.plan(state=state, intent=intent, relationship_profile=None, mood=None, state_delta=delta)

    assert plan.action == Action.ANSWER
    assert "current time" in plan.conversational_goal.lower()
    assert "do not bring up previous conversation topics" in plan.conversational_goal.lower()
    assert plan.allow_multi_bubble is False


def test_action_planner_handles_ongoing_activity_without_past_tense():
    state = ConversationState(conversation_id="test_ongoing")
    intent = SocialIntentAnalyzer.analyze("Naa 7 baje hogi")
    assert intent.social_act == "schedule_future_completion"

    delta = compute_state_delta("Naa 7 baje hogi", intent, state)
    plan = ActionPlanner.plan(state=state, intent=intent, relationship_profile=None, mood=None, state_delta=delta)

    assert plan.action == Action.REACT
    assert "never ask how it was in past tense" in plan.conversational_goal.lower()
    assert plan.allow_multi_bubble is False
