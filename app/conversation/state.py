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

    # What Vesper is currently engaged in
    current_topic: str | None = None          # e.g. "board exams", "that movie", "some meme"
    active_joke: str | None = None            # if a joke thread is in progress
    active_story: str | None = None           # if a story/anecdote is being told

    # What's pending/unresolved
    unanswered_question: str | None = None    # a question the contact asked that hasn't been answered yet

    # Recent history metadata
    last_user_intent: str = "other"
    last_user_emotion: str | None = None      # e.g. "amused", "frustrated", "bored"
    recent_subjects: list[str] = Field(default_factory=list)  # last ~5 subjects/topics mentioned
    last_response: str = ""                   # Vesper's last reply text (for continuity checks)

    # Conversational rhythm tracking
    consecutive_turns: int = 0
    consecutive_short_replies: int = 0        # how many back-to-back ≤3-word replies have been made
    energy_level: float = 0.5                 # internal energy tracker for pacing

    # Strategy tracking
    last_social_act: str = "other"
    last_response_strategy: str = "direct_answer"
    recent_acts: list[str] = Field(default_factory=list)


def _extract_topic(text: str) -> str | None:
    """Naive topic extraction from the incoming message (single pass)."""
    text_lower = text.lower().strip()
    # Academic
    if any(w in text_lower for w in ["exam", "test", "padhai", "padh", "study", "hw", "homework", "maths", "physics", "chemistry"]):
        return "academics"
    # Social
    if any(w in text_lower for w in ["riya", "friend", "yaar", "party", "outing", "bunk"]):
        return "social"
    # Entertainment
    if any(w in text_lower for w in ["reel", "song", "movie", "web series", "spotify", "kya dekh", "kya sun"]):
        return "entertainment"
    # Food
    if any(w in text_lower for w in ["kha", "khana", "biryani", "pizza", "bhooka", "bhukkad"]):
        return "food"
    # Family
    if any(w in text_lower for w in ["mummy", "papa", "bhai", "ghar", "family"]):
        return "family"
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


def update_conversation_state(
    state: ConversationState,
    intent: SocialIntent,
    strategy: str,
    vesper_reply: str = "",
) -> ConversationState:
    """Update dynamic conversation state given incoming intent and chosen strategy."""
    act = intent.social_act
    incoming_text = getattr(intent, "_raw_text", "")  # available if analyzer stores it

    recent = list(state.recent_acts)
    recent.append(act)
    if len(recent) > 7:
        recent = recent[-7:]

    # Recent subjects
    new_topic = _extract_topic(incoming_text) if incoming_text else None
    recent_subjects = list(state.recent_subjects)
    if new_topic and (not recent_subjects or recent_subjects[-1] != new_topic):
        recent_subjects.append(new_topic)
    if len(recent_subjects) > 5:
        recent_subjects = recent_subjects[-5:]

    # Resolve active social mode
    if act in ("playful_insult", "teasing", "reaction_laugh", "humor_attempt"):
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
        energy = max(state.energy_level - 0.1, 0.2)
    else:
        social_mode = state.social_mode  # preserve current mode for continuity
        energy = state.energy_level

    # Track consecutive short replies from Vesper
    vesper_word_count = len(vesper_reply.split()) if vesper_reply else 0
    consec_short = state.consecutive_short_replies
    if vesper_word_count <= 3 and vesper_reply:
        consec_short += 1
    else:
        consec_short = 0

    # Active joke persistence — clear if topic changed significantly
    active_joke = state.active_joke
    if act in ("reaction_laugh",) and state.active_joke:
        active_joke = state.active_joke  # keep it alive
    elif act in ("farewell", "question_logistical", "venting"):
        active_joke = None  # topic shift clears active joke

    # Unanswered question — clear when Vesper replies
    unanswered = state.unanswered_question
    if vesper_reply:
        unanswered = None  # reply was sent, question answered

    # If user asked a question, register it as unanswered (simple heuristic: ends with ?)
    incoming_stripped = incoming_text.strip() if incoming_text else ""
    if incoming_stripped.endswith("?") and not vesper_reply:
        unanswered = incoming_stripped[:80]  # store truncated version

    return ConversationState(
        conversation_id=state.conversation_id,
        contact_id=state.contact_id,
        social_mode=social_mode,
        current_topic=new_topic or state.current_topic,
        active_joke=active_joke,
        active_story=state.active_story,
        unanswered_question=unanswered,
        last_user_intent=act,
        last_user_emotion=_infer_user_emotion(intent),
        recent_subjects=recent_subjects,
        last_response=vesper_reply,
        consecutive_turns=state.consecutive_turns + 1,
        consecutive_short_replies=consec_short,
        energy_level=round(energy, 2),
        last_social_act=act,
        last_response_strategy=strategy,
        recent_acts=recent,
    )
