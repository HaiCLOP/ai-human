"""Conversation state definitions, status enum, composite fingerprint calculation,
and dynamic conversational state tracking.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from app.conversation.social_intent import SocialIntent


class MessageStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    SENT = "SENT"
    FAILED = "FAILED"
    IGNORED = "IGNORED"


def compute_message_fingerprint(
    conversation_id: str,
    sender_handle: str,
    message_text: str,
    timestamp: datetime | None = None,
) -> str:
    """Compute deterministic composite fingerprint for message idempotency.
    
    Hash = SHA256(conversation_id || sender_handle || normalized_text || hour_bucket)
    """
    ts = timestamp or datetime.now(timezone.utc)
    hour_bucket = ts.strftime("%Y-%m-%d-%H")

    # Normalize text by lowercasing and stripping duplicate whitespaces
    normalized_text = re.sub(r"\s+", " ", message_text.strip().lower())

    payload = f"{conversation_id}:{sender_handle}:{normalized_text}:{hour_bucket}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConversationState(BaseModel):
    """Dynamic state of an active conversation — tracks topic, mode, and pending threads."""

    conversation_id: str
    contact_id: str = "default"

    # Active social mode
    social_mode: str = "casual"  # "banter", "casual", "serious", "venting", "logistical", "low_energy"

    # Topic tracking — hierarchical stack (most recent = last item)
    topic_stack: list[str] = Field(default_factory=list)   # e.g. ["school", "exam", "physics stress"]
    current_topic: str | None = None
    previous_topic: str | None = None
    topic_turn_age: int = 0                  # how many turns have passed since current_topic was active/mentioned

    # What Vesper is currently engaged in
    active_joke: str | None = None           # if a joke thread is in progress
    active_story: str | None = None          # if a story/anecdote is being told

    # Question lifecycle tracking
    unanswered_question: str | None = None   # question user asked that hasn't been answered
    questions_asked_by_vesper: list[str] = Field(default_factory=list)   # rolling last 5
    questions_answered_by_user: dict[str, str] = Field(default_factory=dict)  # {normalized_q: answer}
    question_streak: int = 0                 # consecutive turns Vesper asked ≥1 question
    in_interview_mode: bool = False          # True when question_streak >= 3

    # Explicit message tracking (required by contextual resolver)
    last_vesper_message: str = ""            # Vesper's last sent reply
    last_user_message: str = ""             # User's last message (before current)

    # Recent history metadata
    last_user_intent: str = "other"
    last_user_emotion: str | None = None    # e.g. "amused", "frustrated", "bored"
    recent_subjects: list[str] = Field(default_factory=list)  # last ~5 subjects/topics mentioned

    # Conversational rhythm tracking
    consecutive_turns: int = 0
    consecutive_short_replies: int = 0      # how many back-to-back ≤3-word Vesper replies
    energy_level: float = 0.5

    # Conversational effort level
    current_effort_level: str = "NORMAL"   # MINIMAL | LOW | NORMAL | HIGH

    # Strategy tracking (anti-repetition)
    last_social_act: str = "other"
    last_response_strategy: str = "direct_answer"
    recent_acts: list[str] = Field(default_factory=list)
    recent_vesper_strategies: list[str] = Field(default_factory=list)   # last 5 strategies


def _extract_topic(text: str) -> str | None:
    """Topic extraction from the incoming message."""
    text_lower = text.lower().strip()
    # Academic
    if any(w in text_lower for w in ["exam", "test", "padhai", "padh", "study", "hw", "homework", "maths", "physics", "chemistry", "tuition", "marks", "board"]):
        return "academics"
    # Social
    if any(w in text_lower for w in ["riya", "friend", "yaar", "party", "outing", "bunk", "dost"]):
        return "social"
    # Entertainment (movies, films, series, songs)
    if any(w in text_lower for w in ["reel", "song", "movie", "film", "web series", "series", "spotify", "kya dekh", "kya sun", "ok jaanu", "cinema"]):
        return "entertainment"
    # Food
    if any(w in text_lower for w in ["kha", "khana", "biryani", "pizza", "bhooka", "bhukkad", "snack"]):
        return "food"
    # Family
    if any(w in text_lower for w in ["mummy", "papa", "bhai", "ghar", "family"]):
        return "family"
    # Personal status check-in
    if any(w in text_lower for w in ["tum kya kar", "tu kya kar", "kya kar rahi", "kya kar raha", "kya chal raha", "whats up", "what are you doing"]):
        return "personal_status"
    # Stress / Mental break
    if any(w in text_lower for w in ["mental break", "stress", "boards", "tension", "off ho gaya", "tired"]):
        return "stress"
    # Logistical / time
    if any(w in text_lower for w in ["time", "kitne baje", "kaha milna", "kab"]):
        return "logistical"
    return None


def _infer_user_emotion(intent: SocialIntent) -> str | None:
    """Infer the emotional state of the user from their social intent."""
    act = intent.social_act
    if intent.playfulness >= 0.7:
        return "amused"
    if intent.hostility >= 0.6:
        return "hostile"
    if intent.seriousness >= 0.7:
        return "serious"
    if act == "venting":
        return "frustrated"
    if act in ("reaction_laugh", "humor_attempt"):
        return "amused"
    if act in ("greeting", "check_in"):
        return "neutral"
    if act == "farewell":
        return "disengaging"
    return None


def _normalize_question(text: str) -> str:
    """Normalize a question string for registry lookup."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())[:60]


def _vesper_asked_question(reply: str) -> str | None:
    """Extract question from Vesper's reply if one exists."""
    if "?" in reply:
        # Find the question clause
        sentences = re.split(r"[.!]", reply)
        for s in sentences:
            if "?" in s and s.strip():
                return s.strip()[:80]
    return None


def _user_answers_question(user_text: str, question: str) -> bool:
    """Heuristic: does the user message look like an answer to a yes/no or simple question?"""
    lower = user_text.lower().strip()
    if len(lower.split()) <= 6:
        affirmatives = {"haan", "ha", "yes", "yep", "yeah", "ok", "hm", "hmm", "theek", "bilkul", "sahi"}
        negatives = {"nahi", "nope", "no", "na", "nah", "mat"}
        return any(w in lower for w in affirmatives | negatives)
    return True  # longer response almost certainly is an answer


def _compute_effort(user_text: str, intent_act: str) -> str:
    """Determine conversational effort level based on user message."""
    words = len(user_text.split())
    if words <= 2:
        return "MINIMAL"
    if words <= 6:
        return "LOW"
    if intent_act in ("venting",) or words > 20:
        return "HIGH"
    return "NORMAL"


_TOPIC_SHIFT_SIGNALS = {"waise", "btw", "ek aur", "by the way", "alag baat", "topic change", "suno ek baat"}


def update_conversation_state(
    state: ConversationState,
    intent: SocialIntent,
    strategy: str,
    vesper_reply: str = "",
    current_user_text: str = "",
) -> ConversationState:
    """Update dynamic conversation state given incoming intent and chosen strategy."""
    act = intent.social_act
    incoming_text = current_user_text or getattr(intent, "_raw_text", "")

    # --- Recent acts ring buffer ---
    recent = list(state.recent_acts)
    recent.append(act)
    if len(recent) > 7:
        recent = recent[-7:]

    # --- Strategy ring buffer ---
    recent_strategies = list(state.recent_vesper_strategies)
    if strategy:
        recent_strategies.append(strategy)
    if len(recent_strategies) > 5:
        recent_strategies = recent_strategies[-5:]

    # --- Topic stack management & aging ---
    topic_stack = list(state.topic_stack)
    new_topic = _extract_topic(incoming_text) if incoming_text else None
    previous_topic = state.current_topic

    # Check for explicit topic shift signal
    lower_incoming = incoming_text.lower()
    is_topic_shift = any(sig in lower_incoming for sig in _TOPIC_SHIFT_SIGNALS)

    topic_turn_age = state.topic_turn_age

    if is_topic_shift and new_topic:
        topic_stack = [new_topic]   # reset stack on explicit shift
        current_topic = new_topic
        topic_turn_age = 0
    elif new_topic:
        if new_topic != state.current_topic:
            topic_stack.append(new_topic)
            if len(topic_stack) > 5:
                topic_stack = topic_stack[-5:]
        current_topic = new_topic
        topic_turn_age = 0
    else:
        # No topic mentioned in this incoming turn
        topic_turn_age += 1
        if topic_turn_age >= 2:
            # Active topic has expired / cooled down
            current_topic = None
            topic_stack.clear()
        else:
            current_topic = topic_stack[-1] if topic_stack else state.current_topic

    # --- Recent subjects ring buffer ---
    recent_subjects = list(state.recent_subjects)
    if new_topic and (not recent_subjects or recent_subjects[-1] != new_topic):
        recent_subjects.append(new_topic)
    if len(recent_subjects) > 5:
        recent_subjects = recent_subjects[-5:]

    # --- Question registry ---
    questions_asked = list(state.questions_asked_by_vesper)
    questions_answered = dict(state.questions_answered_by_user)

    # Did Vesper ask a question in her last reply?
    vesper_question = _vesper_asked_question(state.last_vesper_message) if state.last_vesper_message else None

    # Did the user answer Vesper's pending question?
    if vesper_question and incoming_text:
        norm_q = _normalize_question(vesper_question)
        if norm_q not in questions_answered and _user_answers_question(incoming_text, vesper_question):
            questions_answered[norm_q] = incoming_text[:100]

    # Register the new Vesper question (if she asked one in THIS reply)
    new_vesper_question = _vesper_asked_question(vesper_reply) if vesper_reply else None
    if new_vesper_question:
        norm_new_q = _normalize_question(new_vesper_question)
        if norm_new_q not in questions_asked:
            questions_asked.append(norm_new_q)
        if len(questions_asked) > 5:
            questions_asked = questions_asked[-5:]

    # --- Question streak / interview mode ---
    if new_vesper_question:
        question_streak = state.question_streak + 1
    else:
        question_streak = 0
    in_interview_mode = question_streak >= 3

    # --- Unanswered question from USER ---
    unanswered = state.unanswered_question
    if vesper_reply:
        unanswered = None   # Vesper replied → question addressed
    incoming_stripped = incoming_text.strip()
    if incoming_stripped.endswith("?") and not vesper_reply:
        unanswered = incoming_stripped[:80]

    # --- Social mode ---
    if act in ("playful_insult", "teasing", "reaction_laugh", "humor_attempt", "counter_tease"):
        social_mode = "banter"
        energy = min(state.energy_level + 0.15, 0.9)
    elif act == "venting":
        social_mode = "venting"
        energy = max(state.energy_level - 0.1, 0.3)
    elif act in ("urgency", "serious_insult"):
        social_mode = "serious"
        energy = 0.8
    elif act in ("question_logistical",):
        social_mode = "logistical"
        energy = 0.5
    elif act in ("acknowledgment", "farewell"):
        social_mode = "low_energy"
        energy = max(state.energy_level - 0.05, 0.2)
    else:
        social_mode = state.social_mode
        energy = state.energy_level

    # --- Consecutive short replies ---
    vesper_word_count = len(vesper_reply.split()) if vesper_reply else 0
    consec_short = state.consecutive_short_replies
    if vesper_word_count <= 3 and vesper_reply:
        consec_short += 1
    else:
        consec_short = 0

    # --- Active joke persistence ---
    active_joke = state.active_joke
    if act in ("reaction_laugh",) and state.active_joke:
        active_joke = state.active_joke
    elif act in ("farewell", "question_logistical", "venting"):
        active_joke = None

    # --- Effort level ---
    effort = _compute_effort(incoming_text, act)

    return ConversationState(
        conversation_id=state.conversation_id,
        contact_id=state.contact_id,
        social_mode=social_mode,
        topic_stack=topic_stack,
        current_topic=current_topic,
        previous_topic=previous_topic,
        topic_turn_age=topic_turn_age,
        active_joke=active_joke,
        active_story=state.active_story,
        unanswered_question=unanswered,
        questions_asked_by_vesper=questions_asked,
        questions_answered_by_user=questions_answered,
        question_streak=question_streak,
        in_interview_mode=in_interview_mode,
        last_vesper_message=vesper_reply or state.last_vesper_message,
        last_user_message=incoming_text or state.last_user_message,
        last_user_intent=act,
        last_user_emotion=_infer_user_emotion(intent),
        recent_subjects=recent_subjects,
        consecutive_turns=state.consecutive_turns + 1,
        consecutive_short_replies=consec_short,
        energy_level=round(energy, 2),
        current_effort_level=effort,
        last_social_act=act,
        last_response_strategy=strategy,
        recent_acts=recent,
        recent_vesper_strategies=recent_strategies,
    )
