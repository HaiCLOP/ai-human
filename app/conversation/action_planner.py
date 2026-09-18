"""Action Planner — determines WHAT Vesper should do before LLM generation.

Runs before PromptBuilder to give the model a clear directive instead of
leaving it to figure out the social situation from raw context alone.

v2: Now takes StateDelta as primary input for context-aware planning.
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
    EXPLAIN = "EXPLAIN"            # Explain Vesper's previous statement (user probed why)
    ELABORATE = "ELABORATE"        # Add more to what Vesper just said (user reacted briefly)
    PLAYFUL_COUNTER = "PLAYFUL_COUNTER"
    TEASE = "TEASE"
    COUNTER_TEASE = "COUNTER_TEASE"  # User counter-teased; Vesper escalates or accepts
    CONTINUE_JOKE = "CONTINUE_JOKE"
    ASK_BACK = "ASK_BACK"
    REACT = "REACT"
    SUPPORT = "SUPPORT"
    CONTINUE_STORY = "CONTINUE_STORY"
    CONTINUE_TOPIC = "CONTINUE_TOPIC"   # Carry active topic forward naturally
    SELF_DISCLOSE = "SELF_DISCLOSE"     # Share something from Vesper's own state/life
    OBSERVE = "OBSERVE"                 # Drop a dry observation
    SEND_REEL = "SEND_REEL"
    WAIT = "WAIT"


@dataclass
class PlannedAction:
    action: Action
    confidence: float                    # 0.0 – 1.0
    conversational_goal: str             # Plain English goal for the LLM (what to accomplish)
    allow_multi_bubble: bool = True      # Whether splitting into 2–3 bubbles is appropriate
    brevity_target: str = "short"        # "very_short", "short", "medium", "long"


# Import here to avoid circular imports at module level
def _get_delta(delta_obj):
    """Duck-type access to StateDelta or a plain dict."""
    if delta_obj is None:
        return None
    if hasattr(delta_obj, "vesper_should_explain"):
        return delta_obj
    return None


class ActionPlanner:
    """Maps social intent + state delta + mood to a concrete action directive."""

    @staticmethod
    def plan(
        state: ConversationState,
        intent: SocialIntent,
        relationship_profile: dict | None = None,
        mood: MoodVector | None = None,
        state_delta=None,  # StateDelta — imported lazily to avoid circular imports
    ) -> PlannedAction:
        """Determine the best action for the current conversational moment."""

        act = intent.social_act
        rel = relationship_profile or {}
        mood = mood or MoodVector()
        delta = _get_delta(state_delta)

        playfulness = rel.get("playfulness", 0.5)
        rel_stage = rel.get("relationship_stage", "friend")
        is_close = rel_stage in ("close_friend", "romantic_relationship", "friend")

        social_energy = mood.social_energy if mood else 0.6
        irritation = mood.irritation if mood else 0.0
        energy = mood.energy if mood else 0.7

        # ═══════════════════════════════════════════════════════
        # PRIORITY 1 — STATE DELTA SIGNALS (trump everything else)
        # ═══════════════════════════════════════════════════════

        if delta:
            # User asked WHY Vesper said something → Vesper must explain
            if delta.vesper_should_explain:
                prev_snippet = state.last_vesper_message[:50] if state.last_vesper_message else "what you just said"
                return PlannedAction(
                    action=Action.EXPLAIN,
                    confidence=0.92,
                    conversational_goal=(
                        f"User asked why you said: '{prev_snippet}'. "
                        "Explain your previous statement naturally in 1-2 short sentences. "
                        "Stay in context — do not change topic, do not ask anything back."
                    ),
                    allow_multi_bubble=True,
                    brevity_target="short",
                )

            # User gave brief reaction to Vesper's substantive statement → Vesper elaborates
            if delta.vesper_should_elaborate:
                prev_snippet = state.last_vesper_message[:50] if state.last_vesper_message else "what you just said"
                return PlannedAction(
                    action=Action.ELABORATE,
                    confidence=0.85,
                    conversational_goal=(
                        f"User reacted briefly to: '{prev_snippet}'. "
                        "Add a bit more to what you said — a detail, a reaction, or a continuation. "
                        "Keep it natural. 1-2 bubbles max."
                    ),
                    allow_multi_bubble=True,
                    brevity_target="short",
                )

            # User answered Vesper's question → acknowledge and advance
            if delta.user_answering_vesper and state.questions_asked_by_vesper:
                return PlannedAction(
                    action=Action.CONTINUE_TOPIC,
                    confidence=0.80,
                    conversational_goal=(
                        "User answered your question. Acknowledge their answer with full context and continue the conversation naturally in complete sentences. "
                        "Always finish your thought — never send an incomplete or cut-off fragment. "
                        "Do NOT repeat the question. Do NOT ask the same thing again."
                    ),
                    allow_multi_bubble=True,
                    brevity_target="medium",
                )


        # ═══════════════════════════════════════════════════════
        # PRIORITY 2 — INTERVIEW MODE GUARD (hard block on questions)
        # ═══════════════════════════════════════════════════════

        if state.in_interview_mode:
            # Force a non-question response — react, self-disclose, or observe
            if energy >= 0.5:
                return PlannedAction(
                    action=Action.SELF_DISCLOSE,
                    confidence=0.85,
                    conversational_goal=(
                        "You've been asking too many questions in a row. "
                        "Don't ask anything this turn. Instead, share something from your own life, "
                        "drop an observation, or react to what's happening. Keep it casual."
                    ),
                    allow_multi_bubble=True,
                    brevity_target="short",
                )
            else:
                return PlannedAction(
                    action=Action.OBSERVE,
                    confidence=0.80,
                    conversational_goal=(
                        "You've been asking too many questions. Don't ask anything. "
                        "Make a short observation or dry comment instead."
                    ),
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )

        # ═══════════════════════════════════════════════════════
        # PRIORITY 3 — "OTHER" / LOW CONFIDENCE → CONTINUE CONTEXT
        # ═══════════════════════════════════════════════════════

        if act == "other" and intent.confidence <= 0.55:
            topic_hint = f" on topic '{state.current_topic}'" if state.current_topic else ""
            last_hint = f" Your last message was: '{state.last_vesper_message[:40]}'." if state.last_vesper_message else ""
            return PlannedAction(
                action=Action.CONTINUE_TOPIC,
                confidence=0.65,
                conversational_goal=(
                    f"Continue the conversation naturally{topic_hint}.{last_hint} "
                    "Respond to what the user just said — don't ignore it, don't change the subject."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # ═══════════════════════════════════════════════════════
        # PRIORITY 4 — SPECIFIC INTENT CASES
        # ═══════════════════════════════════════════════════════

        # ASK_REASON / FOLLOW_UP — context-aware probes
        if act in ("ASK_REASON",):
            topic_hint = f" about '{state.current_topic}'" if state.current_topic else ""
            last_hint = f" User is asking about what you said: '{state.last_vesper_message[:45]}'." if state.last_vesper_message else ""
            return PlannedAction(
                action=Action.EXPLAIN,
                confidence=0.88,
                conversational_goal=(
                    f"User wants to know why/how{topic_hint}.{last_hint} "
                    "Explain briefly and naturally. Stay on topic."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        if act in ("FOLLOW_UP", "REACT_TO_PREVIOUS"):
            topic_hint = f"'{state.current_topic}'" if state.current_topic else "what you were just talking about"
            return PlannedAction(
                action=Action.CONTINUE_TOPIC,
                confidence=0.85,
                conversational_goal=(
                    f"User is following up on {topic_hint}. Continue naturally. "
                    "Don't restart the topic. Don't ask what they mean — just continue."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # Banter / Playful Insult
        if act in ("playful_insult", "counter_tease"):
            if is_close and intent.playfulness >= 0.6 and irritation < 0.5:
                return PlannedAction(
                    action=Action.PLAYFUL_COUNTER,
                    confidence=0.90,
                    conversational_goal=(
                        "User is being playfully insulting or teasing. Counter with a short, confident, "
                        "casually dismissive comeback. No explanations. Max 1 sentence."
                    ),
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )
            elif irritation >= 0.5:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.75,
                    conversational_goal="User is being insulting. Give a flat, unbothered response. Short and dry.",
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )

        # Teasing / Faux Drama
        if act == "teasing":
            if is_close and social_energy >= 0.5:
                return PlannedAction(
                    action=Action.TEASE,
                    confidence=0.85,
                    conversational_goal=(
                        "User is teasing, acting dramatic, or mock-complaining (e.g. saying you rejected them or they are angry). "
                        "Tease back playfully with dry Delhi humor (e.g. 'dramebaaz', 'drama band kar apna 😂', 'itna jaldi gussa ho gaya?'). "
                        "NEVER apologize seriously, NEVER act defensive, NEVER say 'pressure mat de' or 'normal baat kar'. "
                        "Keep it 1 single, punchy bubble."
                    ),
                    allow_multi_bubble=False,
                    brevity_target="short",
                )

        # Active joke continuation
        if state.active_joke and act in ("reaction_laugh", "humor_attempt", "acknowledgment"):
            return PlannedAction(
                action=Action.CONTINUE_JOKE,
                confidence=0.80,
                conversational_goal=(
                    f"There's an ongoing joke: '{state.active_joke[:60]}'. "
                    "Continue or escalate it naturally. Don't explain the joke."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # Laughter / Funny reaction
        if act == "reaction_laugh":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.85,
                conversational_goal=(
                    "User is laughing or reacting to something funny. React naturally — "
                    "agree, add to the joke, or throw in an absurd follow-up. Keep it brief."
                ),
                allow_multi_bubble=True,
                brevity_target="very_short",
            )

        # EXCITEMENT — OMGGG, wait WHAT
        if act == "EXCITEMENT":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.85,
                conversational_goal=(
                    "User sent an excited/surprised reaction. Match the energy — "
                    "short, punchy, curious. Ask what happened or react dramatically."
                ),
                allow_multi_bubble=True,
                brevity_target="very_short",
            )

        # BOREDOM / BORING DAY
        if act == "BOREDOM":
            return PlannedAction(
                action=Action.SELF_DISCLOSE,
                confidence=0.82,
                conversational_goal=(
                    "User is bored or saying it was a boring day. Relate with authentic Delhi teen humor and relatable substance "
                    "(e.g. 'us moment, mera bhi dimaag fry ho gaya aaj', or tease 'kya hua esa? pure din bistar pe pada raha kya 😂', "
                    "or give a real take 'toh room se bahar nikal na thoda, bistar me sadne se bore hi hoga'). "
                    "Have a real opinion or relatable reaction. NEVER interrogate them with multiple vague questions like "
                    "'waah boring day? koi reel dekha? tu bata kya kiya?'. Keep it 1 single, punchy bubble."
                ),
                allow_multi_bubble=False,
                brevity_target="short",
            )

        # Emotional disclosure
        if act == "emotional_disclosure":
            return PlannedAction(
                action=Action.SUPPORT,
                confidence=0.82,
                conversational_goal=(
                    "User is sharing something personal. Acknowledge it, don't lecture. "
                    "Be a friend. Ask a follow-up if appropriate but don't push."
                ),
                allow_multi_bubble=True,
                brevity_target="medium",
            )

        # Direct question
        if act in ("question_personal", "question_logistical", "question_opinion"):
            return PlannedAction(
                action=Action.ANSWER,
                confidence=0.88,
                conversational_goal=(
                    "User asked a direct question. Answer it naturally and briefly. "
                    "You can optionally ask something back if it feels natural, but don't force it."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # Venting
        if act == "venting":
            return PlannedAction(
                action=Action.SUPPORT,
                confidence=0.82,
                conversational_goal=(
                    "User is venting or frustrated. Acknowledge their feeling first, then respond. "
                    "Don't lecture or give unsolicited advice. Be a friend, not a therapist."
                ),
                allow_multi_bubble=True,
                brevity_target="medium",
            )

        # Humor attempt
        if act == "humor_attempt":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.75,
                conversational_goal=(
                    "User made a joke or witty remark. React genuinely — laugh if it's funny, "
                    "give a dry response if it isn't. No forced laughter."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # Check-in / Greeting
        if act in ("greeting", "check_in"):
            if state.consecutive_turns == 0:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.80,
                    conversational_goal="First message or greeting. Respond casually, ask what's up. Not eager.",
                    allow_multi_bubble=True,
                    brevity_target="short",
                )
            else:
                return PlannedAction(
                    action=Action.DIRECT_REPLY,
                    confidence=0.75,
                    conversational_goal="Casual check-in mid-conversation. Short and natural.",
                    allow_multi_bubble=False,
                    brevity_target="very_short",
                )

        # Farewell
        if act == "farewell":
            return PlannedAction(
                action=Action.DIRECT_REPLY,
                confidence=0.90,
                conversational_goal=(
                    "User is leaving. Send a casual, natural send-off. "
                    "Examples: 'haan jaa', 'chal theek hai', 'padh le fir', 'kal baat'. "
                    "NEVER use SILENCE for a farewell."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # Acknowledgment
        if act == "acknowledgment":
            return PlannedAction(
                action=Action.REACT,
                confidence=0.78,
                conversational_goal=(
                    "Very short acknowledgment from user. Match their low effort. "
                    "A single short, natural reaction — 1-5 words max."
                ),
                allow_multi_bubble=False,
                brevity_target="very_short",
            )

        # Story continuation
        if state.active_story:
            return PlannedAction(
                action=Action.CONTINUE_STORY,
                confidence=0.70,
                conversational_goal=f"Continue the story in progress: '{state.active_story[:60]}'. Keep it conversational.",
                allow_multi_bubble=True,
                brevity_target="medium",
            )

        # Unanswered question from user
        if state.unanswered_question and act not in ("farewell",):
            return PlannedAction(
                action=Action.ANSWER,
                confidence=0.65,
                conversational_goal=(
                    f"There was an unanswered question: '{state.unanswered_question[:60]}'. "
                    "Address it while responding to the current message."
                ),
                allow_multi_bubble=True,
                brevity_target="short",
            )

        # Default fallback
        return PlannedAction(
            action=Action.DIRECT_REPLY,
            confidence=0.60,
            conversational_goal=(
                "Respond with real personality, opinions, or witty relatable reactions. "
                "NEVER send vague filler questions or interrogation bubbles (do NOT ask 'waah boring day? koi reel dekha? tu bata kya kiya?'). "
                "Express a real viewpoint. Keep it 1 single, punchy bubble."
            ),
            allow_multi_bubble=False,
            brevity_target="short",
        )
