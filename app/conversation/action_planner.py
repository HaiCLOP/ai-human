"""Action Planner — determines WHAT Vesper should do before LLM generation.

Runs before PromptBuilder to give the model a clear directive instead of
leaving it to figure out the social situation from raw context alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.social_intent import SocialIntent
from app.conversation.state import ConversationState
from app.routine.models import MoodVector


class Action(str, Enum):
    DIRECT_REPLY = "DIRECT_REPLY"
    ANSWER = "ANSWER"
    PLAYFUL_COUNTER = "PLAYFUL_COUNTER"
    TEASE = "TEASE"
    CONTINUE_JOKE = "CONTINUE_JOKE"
    ASK_BACK = "ASK_BACK"
    REACT = "REACT"
    SUPPORT = "SUPPORT"
    CONTINUE_STORY = "CONTINUE_STORY"
    SEND_REEL = "SEND_REEL"
    WAIT = "WAIT"


@dataclass
class PlannedAction:
    action: Action
    confidence: float                  # 0.0 – 1.0
    tactical_hint: str                 # One-line instruction for the LLM
    allow_multi_bubble: bool = True    # Whether splitting into 2–3 bubbles is appropriate
    brevity_target: str = "short"      # "very_short", "short", "medium", "long"


class ActionPlanner:
    """Maps social intent + state + mood to a concrete action directive."""

    @staticmethod
    def plan(
        state: ConversationState,
        intent: SocialIntent,
        relationship_profile: dict | None = None,
        mood: MoodVector | None = None,
    ) -> PlannedAction:
        """Determine the best action for the current conversational moment."""

        act = intent.social_act
        rel = relationship_profile or {}
        mood = mood or MoodVector()

        playfulness = rel.get("playfulness", 0.5)
        rel_stage = rel.get("relationship_stage", "friend")
        is_close = rel_stage in ("close_friend", "romantic_relationship", "friend")
        
        social_energy = mood.social_energy if mood else 0.6
        irritation = mood.irritation if mood else 0.0

        # --- Banter / Playful Insult ---
        if act == "playful_insult":
            if is_close and intent.playfulness >= 0.6 and irritation < 0.5:
                return PlannedAction(
                    action=Action.PLAYFUL_COUNTER,
                    confidence=0.90,
                    tactical_hint=(
                        "The user is playfully insulting you. Counter with a short, confident, casually dismissive comeback. "
                        "No explanations, no defensiveness. Match or slightly escalate the playfulness. Max 1 sentence."
                    ),
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )
            elif irritation >= 0.5:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.75,
                    tactical_hint=(
                        "User is being insulting but you're a bit irritated. Give a flat, unbothered response. Short and dry."
                    ),
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )

        # --- Teasing ---
        if act == "teasing":
            if is_close and social_energy >= 0.5:
                return PlannedAction(
                    action=Action.TEASE,
                    confidence=0.85,
                    tactical_hint=(
                        "User is teasing you. Tease back — match the energy, keep it casual and self-assured. "
                        "Don't over-explain. Short and natural."
                    ),
                    allow_multi_bubble=True,
                    brevity_target="short",
                )

        # --- Active joke continuation ---
        if state.active_joke and act in ("reaction_laugh", "humor_attempt", "acknowledgment"):
            return PlannedAction(
                action=Action.CONTINUE_JOKE,
                confidence=0.80,
                tactical_hint=(
                    f"There's an ongoing joke: '{state.active_joke[:60]}'. "
                    "Continue or escalate it naturally. Don't explain the joke."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # --- Laughter / Funny reaction ---
        if act == "reaction_laugh":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.85,
                tactical_hint=(
                    "User is laughing or reacting to something funny. React naturally — maybe add to the joke, "
                    "agree sarcastically, or throw in an absurd follow-up. Keep it brief."
                ),
                allow_multi_bubble=True,
                brevity_target="very_short",
            )

        # --- Direct question needing an answer ---
        if act in ("question_personal", "question_logistical", "question_opinion"):
            return PlannedAction(
                action=Action.ANSWER,
                confidence=0.88,
                tactical_hint=(
                    "User asked a direct question. Answer it naturally and briefly. "
                    "You can optionally ask something back if it feels natural, but don't force it."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # --- Venting / Support needed ---
        if act == "venting":
            return PlannedAction(
                action=Action.SUPPORT,
                confidence=0.82,
                tactical_hint=(
                    "User is venting or frustrated. Acknowledge their feeling first, then respond. "
                    "Don't lecture or give unsolicited advice. Be a friend, not a therapist."
                ),
                allow_multi_bubble=True,
                brevity_target="medium",
            )

        # --- Humor attempt from user ---
        if act == "humor_attempt":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.75,
                tactical_hint=(
                    "User made a joke or a witty remark. React genuinely — laugh if it's funny, "
                    "give a dry response if it isn't. No forced laughter."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # --- Check-in / Greeting ---
        if act in ("greeting", "check_in"):
            if state.consecutive_turns == 0:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.80,
                    tactical_hint="Greeting or first message. Respond casually and warmly (but not eagerly). Ask what's up.",
                    allow_multi_bubble=True,
                    brevity_target="short",
                )
            else:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.75,
                    tactical_hint="Casual check-in mid-conversation. Short and natural.",
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )

        # --- Farewell ---
        if act == "farewell":
            return PlannedAction(
                action=Action.DIRECT_REPLY,
                confidence=0.90,
                tactical_hint=(
                    "User is leaving. Send a casual, natural send-off. "
                    "Examples: 'haan jaa', 'chal theek hai', 'padh le fir', 'kal baat'. "
                    "NEVER use SILENCE for a farewell."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # --- Standalone short acknowledgments (haan, ok, acha, mast, 😭) ---
        if act == "acknowledgment":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.78,
                tactical_hint=(
                    "Very short acknowledgment from user. Match their low effort. "
                    "A single short, natural reaction — 1–5 words max."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # --- Story continuation ---
        if state.active_story:
            return PlannedAction(
                action=Action.CONTINUE_STORY,
                confidence=0.70,
                tactical_hint=f"Continue the story in progress: '{state.active_story[:60]}'. Keep it conversational.",
                allow_multi_bubble=True,
                brevity_target="medium",
            )

        # --- Unanswered question present ---
        if state.unanswered_question and act not in ("farewell",):
            return PlannedAction(
                action=Action.ANSWER,
                confidence=0.65,
                tactical_hint=(
                    f"There was an unanswered question earlier: '{state.unanswered_question[:60]}'. "
                    "If contextually appropriate, address it while responding to the current message."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # --- Default fallback ---
        return PlannedAction(
            action=Action.DIRECT_REPLY,
            confidence=0.60,
            tactical_hint="Respond naturally and conversationally to what the user said. Keep it short and in-character.",
            allow_multi_bubble=True,
            brevity_target="short",
        )
