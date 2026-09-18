"""Pydantic data models for Historical Conversation Intelligence."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Classified types of Instagram messages."""
    TEXT = "TEXT"
    REEL = "REEL"
    STORY = "STORY"
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    REACTION = "REACTION"
    CALL = "CALL"
    ATTACHMENT = "ATTACHMENT"
    UNKNOWN = "UNKNOWN"


class ReactionItem(BaseModel):
    """Reaction attached to an Instagram message."""
    reaction: str
    actor: str


class NormalizedMessage(BaseModel):
    """Normalized internal representation of an individual message."""
    conversation_id: str
    sender_id: str
    sender_display_name: str
    timestamp_ms: int
    timestamp_iso: str
    message_type: MessageType
    text: str = ""
    message_index: int = 0
    sender_role: Literal["operator", "contact", "unknown"] = "unknown"
    reply_to: str | None = None
    edited: bool = False
    shared_url: str | None = None
    share_text: str | None = None
    original_content_owner: str | None = None
    reactions: list[ReactionItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.text

    @property
    def sender(self) -> str:
        return self.sender_role

    @property
    def sender_name(self) -> str:
        return self.sender_display_name


class HistoricalTurn(BaseModel):
    """A contextual turn representing a contact message and the subsequent operator response."""
    turn_id: str
    conversation_id: str
    contact_id: str
    contact_text: str
    operator_text: str
    timestamp_ms: int
    social_act: str = "other"
    social_intent: dict[str, Any] = Field(default_factory=dict)
    response_strategy: str = "direct_answer"
    response_length_category: str = "short"  # "very_short", "short", "medium", "long"
    context_messages: list[dict[str, str]] = Field(default_factory=list)
    turn_energy: float = 0.5


class BehavioralPattern(BaseModel):
    """Statistically derived situation-to-strategy behavioral pattern."""
    pattern_id: str
    contact_id: str = "global"
    trigger_social_act: str
    response_strategy: str
    sample_count: int = 1
    confidence: float = 0.5
    typical_length: str = "short"
    example_pairs: list[dict[str, str]] = Field(default_factory=list)


class NormalizedConversation(BaseModel):
    """Normalized representation of a single conversation thread."""
    conversation_id: str
    file_path: str
    participants: list[str]
    operator_name: str | None = None
    contact_name: str | None = None
    contact_id: str | None = None
    messages: list[NormalizedMessage] = Field(default_factory=list)
    message_count: int = 0
    date_start: str | None = None
    date_end: str | None = None


class DatasetManifest(BaseModel):
    """Metadata summary of a discovered Instagram historical dataset."""
    files_discovered: int = 0
    valid_conversations: int = 0
    invalid_files: int = 0
    total_messages: int = 0
    type_counts: dict[str, int] = Field(default_factory=dict)
    date_range: tuple[str, str] | None = None
    dataset_hash: str = ""
    invalid_file_reasons: dict[str, str] = Field(default_factory=dict)


class StyleTraitObservation(BaseModel):
    """Individual observable trait with statistical evidence and confidence."""
    trait_name: str
    trait_value: str
    confidence: float
    evidence_count: int
    first_seen: str
    last_seen: str


class OperatorStyleProfile(BaseModel):
    """Global learned communication profile of the human operator."""
    version_id: str = "v001"
    total_messages: int = 0
    hinglish_ratio: float = 0.0
    english_ratio: float = 0.0
    hindi_ratio: float = 0.0
    lowercase_ratio: float = 0.0
    ending_period_ratio: float = 0.0
    contraction_apostrophe_ratio: float = 0.0
    avg_message_length_chars: float = 0.0
    median_message_length_chars: float = 0.0
    avg_words_per_message: float = 0.0
    burst_message_ratio: float = 0.0
    emoji_density: float = 0.0
    favorite_emojis: list[tuple[str, int]] = Field(default_factory=list)
    slang_frequencies: dict[str, float] = Field(default_factory=dict)
    question_frequency: float = 0.0
    common_openers: list[str] = Field(default_factory=list)
    recent_style: dict[str, Any] = Field(default_factory=dict)
    long_term_style: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    evidence_count: int = 0
    updated_at: str = ""


class ContactStyleProfile(BaseModel):
    """Learned communication profile of a specific conversation partner."""
    contact_id: str
    display_name: str
    total_messages: int = 0
    brevity_level: str = "medium"
    avg_message_length_chars: float = 0.0
    hinglish_ratio: float = 0.0
    emoji_density: float = 0.0
    sarcasm_score: float = 0.0
    formality_score: float = 0.0
    favorite_emojis: list[tuple[str, int]] = Field(default_factory=list)
    slang_terms_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_count: int = 0
    updated_at: str = ""


class RelationshipStyleProfile(BaseModel):
    """Dyadic relationship dynamics between the operator and a contact."""
    contact_id: str
    playfulness: float = 0.5
    sarcasm: float = 0.3
    teasing: float = 0.3
    seriousness: float = 0.4
    operator_initiation_ratio: float = 0.5
    operator_message_ratio: float = 0.5
    avg_exchange_length: float = 4.0
    reel_sharing_frequency: float = 0.0
    response_style: str = "casual"
    confidence: float = 0.0
    evidence_count: int = 0
    updated_at: str = ""


class HumorProfile(BaseModel):
    """Humor, teasing, and laughing analysis for a contact or globally."""
    contact_id: str = "global"
    humor_frequency: float = 0.0
    sarcasm_frequency: float = 0.0
    teasing_frequency: float = 0.0
    dark_humor_frequency: float = 0.0
    callback_frequency: float = 0.0
    laughing_reactions_count: int = 0
    confidence: float = 0.0
    updated_at: str = ""


class ReelIntelligenceProfile(BaseModel):
    """Reel sharing habits and reaction patterns."""
    total_reels_shared: int = 0
    sent_by_operator_count: int = 0
    received_by_operator_count: int = 0
    shared_with_commentary_ratio: float = 0.0
    top_shared_creators: list[str] = Field(default_factory=list)
    recipient_reaction_rate: float = 0.0
    confidence: float = 0.0


class MemoryCandidate(BaseModel):
    """Candidate memory extracted from conversation history."""
    memory_id: str
    contact_id: str
    category: str
    statement: str
    importance: float = 0.5
    confidence: float = 0.5
    frequency: int = 1
    recency: str = ""
    source_count: int = 1


class ValidatedActionPlan(BaseModel):
    """Strict structured JSON schema governing all character output actions."""
    action: Literal["send_message", "send_reel", "send_message_and_reel", "wait"] = Field(
        default="send_message",
        description="Action chosen by the character.",
    )
    message: str | None = Field(
        default=None,
        description="Text message content to send (if action includes send_message).",
    )
    reel: dict[str, Any] | None = Field(
        default=None,
        description="Reel object containing 'required', 'url', and 'reason' (if action includes send_reel).",
    )
    style_context: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores of applied style systems.",
    )
