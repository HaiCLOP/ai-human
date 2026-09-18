"""Deterministic Conversation Policy and Contribution Planning Layer (Vesper V3).

Separates:
1. WHAT should Vesper contribute? (Deterministic policy deciding conversational move, effort, question budget)
2. HOW should she say it? (LLM realizing the planned contribution into authentic Hinglish)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.conversation.social_intent import SocialIntent
from app.conversation.state import ConversationState
from app.core.logging import get_logger
from app.routine.models import MoodVector, VesperLifeState

logger = get_logger("conversation.policy")


class ContributionType(str, Enum):
    ANSWER_ONLY = "ANSWER_ONLY"
    REACTION_ONLY = "REACTION_ONLY"
    ANSWER_PLUS_REACTION = "ANSWER_PLUS_REACTION"
    COUNTER_TEASE = "COUNTER_TEASE"
    PLAYFUL_DEFLECTION = "PLAYFUL_DEFLECTION"
    SELF_DISCLOSURE = "SELF_DISCLOSURE"
    STORY_DETAIL = "STORY_DETAIL"
    OPINION = "OPINION"
    CALLBACK = "CALLBACK"
    TOPIC_CONTINUATION = "TOPIC_CONTINUATION"
    TOPIC_SHIFT = "TOPIC_SHIFT"
    QUESTION = "QUESTION"
    SUPPORT = "SUPPORT"
    CLOSE = "CLOSE"
    SEND_REEL = "SEND_REEL"
    MESSAGE_AND_REEL = "MESSAGE_AND_REEL"
    WAIT = "WAIT"


class ConversationContributionPlan(BaseModel):
    """Structured plan dictating Vesper's conversational contribution before generation."""

    contribution_type: ContributionType
    should_ask_question: bool = Field(
        default=False,
        description="Whether asking a question is permitted in this turn.",
    )
    question_budget: int = Field(
        default=0,
        description="Exact number of questions permitted (default 0). Never ask questions simply to continue chat.",
    )
    question_reason: str | None = Field(
        default=None,
        description="Conversational rationale if a question is allowed.",
    )
    target_topic: str | None = Field(
        default=None,
        description="The topic to address or shift to.",
    )
    information_to_address: str | None = Field(
        default=None,
        description="What specific information from user message / state to acknowledge or answer.",
    )
    information_to_add: str | None = Field(
        default=None,
        description="Specific fact, opinion, observation, or life state detail to introduce.",
    )
    emotional_direction: str = Field(
        default="neutral",
        description="Emotional tone (e.g. 'deadpan', 'playful_teasing', 'dry_agreement', 'supportive', 'relatable_vent').",
    )
    desired_effort: Literal["very_low", "low", "medium", "high"] = Field(
        default="low",
        description="Effort level matching user conversational investment.",
    )
    desired_length: Literal["very_short", "short", "medium", "long"] = Field(
        default="short",
        description="Target reply length.",
    )
    allow_topic_shift: bool = Field(
        default=False,
        description="Whether shifting away from current topic is permitted.",
    )
    allow_callback: bool = Field(default=True)
    allow_self_disclosure: bool = Field(
        default=False,
        description="Whether volunteering info about Vesper's life is allowed.",
    )
    allow_humor: bool = Field(default=True)
    urgency: float = Field(default=0.1)
    reason: str = Field(
        default="",
        description="Policy reasoning behind this contribution choice.",
    )

    def to_prompt_text(self) -> str:
        """Format the plan as a clean directive section for LLM realization."""
        lines = [
            "[CONTRIBUTION PLAN — WHAT TO CONTRIBUTE]",
            f"Contribution Type: {self.contribution_type.value}",
            f"Conversational Objective: {self.reason}",
        ]
        if self.information_to_address:
            lines.append(f"Information to Address: {self.information_to_address}")
        if self.information_to_add:
            lines.append(f"Information to Add: {self.information_to_add}")

        if self.should_ask_question and self.question_budget > 0:
            lines.append(f"Question Budget: {self.question_budget} (Permitted Reason: {self.question_reason})")
        else:
            lines.append("Question Budget: 0 (STRICT BAN: DO NOT ask any question — no 'tu bata', no interrogation)")

        lines.append(f"Effort Parity: {self.desired_effort} | Desired Length: {self.desired_length}")

        if not self.allow_self_disclosure:
            lines.append("Self-Disclosure: FORBIDDEN (do NOT volunteer what you are doing unless explicitly asked)")
        if self.allow_topic_shift:
            lines.append("Topic Shift: PERMITTED (freely let topic drop or change)")

        return "\n".join(lines)


class ConversationPolicy:
    """Deterministic policy determining the contribution plan before realization."""

    CLOSER_WORDS = {"k", "ok", "okay", "okayy", "okayyy", "haan", "hmm", "theek hai", "chal", "bye", "gn"}
    SHORT_REACTIONS = {"mast", "mast hai", "sahi hai", "sahi", "bekar", "accha", "acha", "nice", "cool", "badhiya", "wah", "lmao", "lol"}

    @classmethod
    def evaluate(
        cls,
        incoming_text: str | None = None,
        intent: SocialIntent | None = None,
        conv_state: ConversationState | None = None,
        life_state: VesperLifeState | None = None,
        relationship: dict[str, Any] | None = None,
        mood: MoodVector | None = None,
        current_dt: datetime | None = None,
        **kwargs: Any,
    ) -> ConversationContributionPlan:
        """Evaluate situation and return authoritative contribution plan."""
        raw_text = (incoming_text or kwargs.get("clean_text") or "").strip()
        lower = raw_text.lower()
        words = lower.split()

        act_intent = intent or kwargs.get("social_intent")
        if act_intent is None:
            act_intent = SocialIntentAnalyzer.analyze(raw_text)
        act = act_intent.social_act

        conv_state = conv_state or kwargs.get("state") or ConversationState(conversation_id="default")
        life_state = life_state or kwargs.get("life_state") or VesperLifeState(
            activity="studying at home",
            location="study desk",
            current_task="homework",
            energy=0.6,
            mood_label="neutral",
            free_time=False,
            next_event="dinner",
            unfinished_thought="assignments to finish",
        )
        relationship = relationship or kwargs.get("rel_style") or {}
        intent = act_intent
        now_dt = current_dt or datetime.now(timezone.utc)

        # 1. LOW-EFFORT CLOSERS & ACKNOWLEDGMENTS ("Okayyy", "haan", "k", "hmm")
        if act in ("acknowledgment", "farewell") or (len(words) <= 2 and lower in cls.CLOSER_WORDS):
            return ConversationContributionPlan(
                contribution_type=ContributionType.CLOSE,
                should_ask_question=False,
                question_budget=0,
                desired_effort="very_low",
                desired_length="very_short",
                allow_self_disclosure=False,
                allow_topic_shift=True,
                information_to_address=f"User acknowledged/closed with '{raw_text}'",
                information_to_add="Ultra-short matching closer ('haan', 'yepp', 'yess', 'hmm', 'chal')",
                reason="Match user's low effort with single-word closer. Strictly no unprompted self-updates.",
            )

        # 2. TOPIC DISMISSAL ("Chhoro", "bhul jao", "rehne de", "drop it")
        if act == "topic_dismissal" or any(w in lower for w in ["chhoro", "chhodo", "chhod na", "rehne de", "rehne do", "bhul jao", "drop it", "leave it"]):
            return ConversationContributionPlan(
                contribution_type=ContributionType.TOPIC_SHIFT,
                should_ask_question=False,
                question_budget=0,
                desired_effort="very_low",
                desired_length="very_short",
                allow_self_disclosure=False,
                allow_topic_shift=True,
                information_to_address="User wants to drop the previous topic",
                information_to_add="Drop topic immediately with zero resistance ('haan chhor', 'chal theek hai', 'aur bata')",
                reason="User explicitly requested topic dismissal ('Chhoro'). Drop subject immediately.",
            )

        # 3. RHETORICAL CHALLENGE ("Toh?", "So?", "So what?")
        if act == "rhetorical_challenge" or lower in ("toh?", "to?", "so?", "so what?", "toh kya?", "fir?"):
            return ConversationContributionPlan(
                contribution_type=ContributionType.PLAYFUL_DEFLECTION,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="very_short",
                allow_self_disclosure=False,
                information_to_address="User challenged pointless detail with 'Toh?'",
                information_to_add="Laugh at yourself for rambling ('kuch nahi bas aise hi bol diya lol', 'haan matlab kuch nahi'). Do NOT elaborate on the story!",
                reason="User challenged a pointless observation ('Toh?'). Deflect lightly without repeating story.",
            )

        # 4. USER CONFUSION / CLARIFICATION ("Kya bol rahi ho", "Samjha nahi", "Heh", "kuch bhi")
        if act in ("clarification", "user_confusion") or any(w in lower for w in ["kya bol rahi", "kya bol rahe", "kuch bhi", "pagal hai kya", "heh", "samjha nahi"]):
            return ConversationContributionPlan(
                contribution_type=ContributionType.PLAYFUL_DEFLECTION,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="short",
                allow_self_disclosure=False,
                information_to_address=f"User expressed confusion at previous message: '{raw_text}'",
                information_to_add="Laugh it off or casually dismiss self ('arre kuch nahi haha', 'chhod yaar dimag fry ho gaya tha'). STRICT BAN: Never repeat the previous statement or story!",
                reason="User is confused. Dismiss lightly, laugh it off, and reset conversational thread.",
            )

        # 5. SHORT REACTION ("Mast hai", "Sahi hai", "Bekar", "Accha")
        if act in ("reaction_short", "reaction_opinion") or (len(words) <= 3 and lower in cls.SHORT_REACTIONS):
            return ConversationContributionPlan(
                contribution_type=ContributionType.REACTION_ONLY,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="very_short",
                allow_self_disclosure=False,
                information_to_address=f"User's short reaction '{raw_text}'",
                information_to_add="Natural brief reaction or opinion ('sahi hai', 'haan dekh le', 'mujhe pata tha mast lagegi'). NO questions!",
                reason="User shared a short reaction ('Mast hai'). Share an opinion or acknowledgment without interrogating.",
            )

        # 6. LOGISTICAL / TIME QUERY ("What's the time", "kitne baje")
        if act == "question_logistical" or any(w in lower for w in ["time", "baje", "ghadi", "clock"]):
            return ConversationContributionPlan(
                contribution_type=ContributionType.ANSWER_ONLY,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="very_short",
                allow_self_disclosure=False,
                information_to_address="Time inquiry",
                information_to_add="Current local time from [NOW]. Pure time answer with zero questions about other topics.",
                reason="Answer logistical time query purely with current time.",
            )

        # 7. ONGOING ACTIVITY & TEMPORAL SCHEDULES ("Naa 7 baje hogi", "abhi dekh raha hu")
        if (
            act in ("schedule_future_completion", "ongoing_activity")
            or conv_state.is_subject_ongoing(conv_state.current_topic or "movie", now_dt)
            or any(re.search(p, lower) for p in [
                r"\b(naa?|nahi|no)?\s*\d+\s*baje\b",
                r"\babhi\s+(dekh|padh|khel|chal)\s+rah",
                r"\bkal\s+(exam|test)",
            ])
        ):
            return ConversationContributionPlan(
                contribution_type=ContributionType.REACTION_ONLY,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="very_short",
                allow_self_disclosure=False,
                information_to_address=f"Ongoing activity/schedule ending in future: '{raw_text}'",
                information_to_add="Acknowledge casually ('achha theek dekh le fir', 'chal baad me batana'). STRICT BAN: Never ask past-tense questions ('kaisi thi', 'khatam ho gayi').",
                reason="Event is ongoing. Acknowledge and allow user to finish without interrogation.",
            )

        # 8. ACTIVITY / STATUS INQUIRY ("wyd", "kya kar rahi ho", "kaha ho", "what happened today")
        if "activity_query_pattern" in intent.detected_signals or any(w in lower for w in [
            "wyd", "kya kar rahi", "kya kar rahe", "kaha ho", "kaha hai",
            "what happened", "what did you do", "kya hua aaj", "din kaisa", "aaj kya kiya"
        ]):
            return ConversationContributionPlan(
                contribution_type=ContributionType.SELF_DISCLOSURE,
                should_ask_question=False,
                question_budget=0,
                desired_effort="medium",
                desired_length="short",
                allow_self_disclosure=True,
                information_to_address="User asked what Vesper is doing",
                information_to_add=(
                    f"Authoritative life state: activity={life_state.activity}, "
                    f"task={life_state.current_task}, location={life_state.location}. "
                    f"Give concrete real detail, NOT generic filler like 'bas chill'."
                ),
                reason="Realize authoritative life state accurately with specific task and location.",
            )

        # 9. EXPLAIN PROBE ("kyu?", "why?")
        if act == "ask_reason" or lower in ("kyu?", "why?", "kyun?", "kyu"):
            return ConversationContributionPlan(
                contribution_type=ContributionType.STORY_DETAIL,
                should_ask_question=False,
                question_budget=0,
                desired_effort="medium",
                desired_length="short",
                allow_self_disclosure=True,
                information_to_address=f"Explain previous statement: '{conv_state.last_vesper_message}'",
                information_to_add=(
                    f"Concrete explanation based on life state task ({life_state.current_task}) "
                    f"or unfinished thought ({life_state.unfinished_thought}). Give actual substance, not vague 'bas'."
                ),
                reason="User asked why. Provide a genuine, grounded explanation.",
            )

        # 10. PLAYFUL INSULT / BANTER ("bro tu pagal hai", "you're dumb")
        if act in ("playful_insult", "teasing"):
            return ConversationContributionPlan(
                contribution_type=ContributionType.COUNTER_TEASE,
                should_ask_question=False,
                question_budget=0,
                desired_effort="low",
                desired_length="very_short",
                allow_humor=True,
                allow_self_disclosure=False,
                information_to_address="User's playful roast/tease",
                information_to_add="Quick witty counter-tease or dry deadpan comeback. NEVER act defensive or preachy.",
                reason="Playful banter. Return an effortless counter-tease.",
            )

        # 11. HIGH-EFFORT VENTING / LONG MESSAGE (> 15 words)
        if act == "venting" or len(words) >= 15:
            return ConversationContributionPlan(
                contribution_type=ContributionType.SUPPORT,
                should_ask_question=True,
                question_budget=1,
                question_reason="Empathetic follow-up question on user's venting/story",
                desired_effort="high",
                desired_length="medium",
                allow_self_disclosure=False,
                information_to_address="User is sharing an emotional vent or detailed story",
                information_to_add="Acknowledge and validate their situation first, then ask 1 gentle follow-up question.",
                reason="User invested high effort. Match effort with supportive engagement.",
            )

        # 12. GENERAL / TOPIC CONTINUATION (Default fallback)
        can_ask = (conv_state.question_streak == 0 and intent.urgency > 0.4)
        q_budget = 1 if can_ask else 0

        return ConversationContributionPlan(
            contribution_type=ContributionType.TOPIC_CONTINUATION,
            should_ask_question=can_ask,
            question_budget=q_budget,
            question_reason="Genuine curiosity on active topic" if can_ask else None,
            target_topic=conv_state.current_topic,
            desired_effort="low",
            desired_length="short",
            allow_self_disclosure=False,
            allow_topic_shift=conv_state.topic_turn_age >= 1,
            information_to_address=f"User statement: '{raw_text}'",
            information_to_add="Contribute a real viewpoint, opinion, or observation. Do NOT force a question.",
            reason="Continue topic naturally with substance, avoiding robotic interrogation.",
        )
