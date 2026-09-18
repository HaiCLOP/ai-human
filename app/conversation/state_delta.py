"""State Delta Calculator — answers 'what just changed?' before action planning.

Computes the conversational delta between the previous state and the current
incoming message. This is the primary input to the ActionPlanner and ensures
Qwen receives the correct conversational situation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.conversation.social_intent import SocialIntent
from app.conversation.state import ConversationState


# Patterns that signal the user is probing Vesper's previous statement
_WHY_PROBE_PATTERNS = [
    r"^(kyuu+\??|kyu\??|why\?*|why tho\??|kyu bata|kyun\?*)$",
    r"^(kyuu+|kyu)\s*\??$",
]

# Patterns that signal continuation/follow-up
_CONTINUATION_PROBE_PATTERNS = [
    r"^(phir\??|then\??|aur\??|then what\??|aur kya\??|kya hua fir\?*|what happened\?*)$",
    r"^(matlab\??|seriously\??|srsly\??|really\?*)$",
]

# Pure reaction patterns (very short, no new info)
_PURE_REACTION_PATTERNS = [
    r"^(ohh+|ooh+|acha+|achha+|hmm+|hm+|okay+|ok+)$",
    r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF]+$",  # emoji-only
    r"^(omg+|omgg+|bhai+|bruh+|wait|wow+)$",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower().strip()
    return any(re.match(p, lower) for p in patterns)


def _is_why_probe(text: str) -> bool:
    return _match_any(text, _WHY_PROBE_PATTERNS)


def _is_continuation_probe(text: str) -> bool:
    return _match_any(text, _CONTINUATION_PROBE_PATTERNS)


def _is_pure_reaction(text: str) -> bool:
    return _match_any(text, _PURE_REACTION_PATTERNS) or (
        len(text.strip().split()) <= 2 and not text.strip().endswith("?")
    )


def _estimate_effort(user_text: str, intent_act: str) -> str:
    words = len(user_text.split())
    if words <= 2:
        return "MINIMAL"
    if words <= 6:
        return "LOW"
    if intent_act == "venting" or words > 20:
        return "HIGH"
    return "NORMAL"


@dataclass
class StateDelta:
    """What changed between the previous state and the current incoming message.

    This is the primary input to the ActionPlanner — not just the intent label.
    """

    # Plain-English description of what happened
    what_changed: str

    # What the user's message is targeting
    target_of_message: str          # "vesper_previous_statement" | "current_topic" | "new_topic" | "user_self" | "unknown"

    # Specific action signals
    vesper_should_explain: bool     # User probed Vesper's prev statement with why/phir
    vesper_should_elaborate: bool   # User reacted briefly; Vesper should add more
    user_answering_vesper: bool     # User is answering Vesper's pending question

    # Topic dynamics
    topic_continued: bool
    topic_shifted: bool
    new_topic_introduced: bool

    # Emotional dynamics
    emotional_shift: str | None     # "escalating_banter" | "de-escalating" | "going_serious" | None

    # Response guidance
    implied_goal: str               # "explain_stress" | "react_to_joke" | "continue_story" | "answer_question" | etc.
    effort_suggested: str           # MINIMAL | LOW | NORMAL | HIGH


def compute_state_delta(
    current_msg: str,
    intent: SocialIntent,
    state: ConversationState,
) -> StateDelta:
    """Compute what changed between previous state and the current message."""

    last_vesper = state.last_vesper_message
    last_vesper_words = len(last_vesper.split()) if last_vesper else 0
    act = intent.social_act

    # --- Determine target ---
    target = "current_topic"
    if last_vesper and last_vesper_words > 3 and _is_why_probe(current_msg):
        target = "vesper_previous_statement"
    elif last_vesper and last_vesper_words > 3 and _is_continuation_probe(current_msg):
        target = "vesper_previous_statement"
    elif last_vesper and _is_pure_reaction(current_msg):
        target = "vesper_previous_statement"
    elif act in ("question_personal", "question_logistical", "ASK_REASON"):
        target = "current_topic"
    elif act == "venting":
        target = "user_self"

    # --- Key boolean signals ---
    vesper_should_explain = (
        target == "vesper_previous_statement"
        and _is_why_probe(current_msg)
        and last_vesper_words > 3
    )

    vesper_should_elaborate = (
        target == "vesper_previous_statement"
        and _is_pure_reaction(current_msg)
        and last_vesper_words > 5
        and not _is_why_probe(current_msg)
    )

    user_answering_vesper = bool(
        state.questions_asked_by_vesper
        and len(current_msg.split()) <= 8
        and act in ("acknowledgment", "agreement", "disagreement", "question_personal", "other",
                    "FOLLOW_UP", "REACT_TO_PREVIOUS", "CONTINUE_TOPIC")
    )

    # --- Topic dynamics ---
    topic_continued = (state.current_topic is not None) and (not topic_shifted_heuristic(current_msg, state))
    topic_shifted = topic_shifted_heuristic(current_msg, state)
    new_topic_introduced = topic_shifted

    # --- Emotional shift ---
    emotional_shift: str | None = None
    if state.social_mode == "banter" and act == "venting":
        emotional_shift = "de-escalating"
    elif state.social_mode == "casual" and act in ("playful_insult", "teasing"):
        emotional_shift = "escalating_banter"
    elif state.social_mode == "banter" and act in ("venting", "serious_insult"):
        emotional_shift = "going_serious"

    # --- Implied goal ---
    if vesper_should_explain:
        implied_goal = f"explain_previous: {last_vesper[:40]}"
    elif vesper_should_elaborate:
        implied_goal = f"elaborate_on: {last_vesper[:40]}"
    elif user_answering_vesper and state.questions_asked_by_vesper:
        implied_goal = f"acknowledge_answer: {state.questions_asked_by_vesper[-1][:40]}"
    elif act == "venting":
        implied_goal = "validate_and_empathize"
    elif act in ("playful_insult", "teasing", "counter_tease"):
        implied_goal = "banter_response"
    elif act == "farewell":
        implied_goal = "casual_sendoff"
    elif act in ("question_personal", "question_logistical"):
        implied_goal = "answer_question"
    elif act in ("reaction_laugh",):
        implied_goal = "react_to_laugh"
    elif act == "acknowledgment" and topic_continued:
        implied_goal = "continue_or_let_breathe"
    else:
        implied_goal = "continue_conversation"

    # --- What changed (human readable) ---
    if vesper_should_explain:
        what_changed = f"User asked why Vesper said: '{last_vesper[:50]}'"
    elif vesper_should_elaborate:
        what_changed = f"User reacted briefly to Vesper's statement; Vesper should continue"
    elif user_answering_vesper:
        what_changed = f"User answered Vesper's question: '{current_msg[:50]}'"
    elif topic_shifted:
        what_changed = f"Topic shifted from '{state.current_topic}' to something new"
    elif act == "venting":
        what_changed = "User is venting/frustrated"
    else:
        what_changed = f"Conversation continued: {act} on topic '{state.current_topic or 'general'}'"

    effort = _estimate_effort(current_msg, act)

    return StateDelta(
        what_changed=what_changed,
        target_of_message=target,
        vesper_should_explain=vesper_should_explain,
        vesper_should_elaborate=vesper_should_elaborate,
        user_answering_vesper=user_answering_vesper,
        topic_continued=topic_continued,
        topic_shifted=topic_shifted,
        new_topic_introduced=new_topic_introduced,
        emotional_shift=emotional_shift,
        implied_goal=implied_goal,
        effort_suggested=effort,
    )


_TOPIC_SHIFT_WORDS = {"waise", "btw", "ek aur", "by the way", "alag baat", "suno"}


def topic_shifted_heuristic(current_msg: str, state: ConversationState) -> bool:
    """Return True if the message appears to shift away from the current topic."""
    lower = current_msg.lower()
    if any(sig in lower for sig in _TOPIC_SHIFT_WORDS):
        return True
    # If there's a current topic and the new message has no keywords related to it
    if state.current_topic and state.current_topic not in lower:
        # Only flag as shift if message is substantive
        if len(current_msg.split()) > 4:
            return True
    return False
