"""Response Intent — structured intermediate output between planning and generation.

Separates classification from wording. ActionPlanner fills this struct;
Qwen generates text to fulfill it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


def _get_ist_zone():
    """Return Asia/Kolkata ZoneInfo if available, else a UTC+5:30 fixed offset."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Kolkata")
    except Exception:
        return timezone(timedelta(hours=5, minutes=30))


@dataclass
class ResponseIntent:
    """What Vesper should do and HOW, before the LLM is invoked."""

    # What Vesper is doing (from ActionPlanner)
    social_act: str                 # What the user was doing
    response_action: str            # What Vesper should do (Action enum value)
    goal: str                       # Plain English goal: "explain your stress"

    # Style guidance
    effort: str = "NORMAL"          # MINIMAL | LOW | NORMAL | HIGH
    tone: str = "neutral"           # "slightly_tired" | "playful" | "neutral" | "warm" | "dry"
    language_hint: str = "hinglish" # "hinglish" | "mostly_hindi" | "mixed" | "mostly_english"
    bubble_count_target: int = 1    # 1 | 2 | 3

    # Guardrails
    allow_question: bool = True     # False when in_interview_mode
    avoid_topics: list[str] = field(default_factory=list)   # saturated topics

    def to_prompt_block(self) -> str:
        """Format as [RESPONSE OBJECTIVE] block for the prompt."""
        lines = [
            f"[RESPONSE OBJECTIVE]",
            f"Action: {self.response_action}",
            f"Goal: {self.goal}",
            f"Effort: {self.effort}",
            f"Tone: {self.tone}",
            f"Language: {self.language_hint}",
            f"Bubbles: {self.bubble_count_target}",
        ]
        if not self.allow_question:
            lines.append("IMPORTANT: Do NOT ask any question this turn.")
        if self.avoid_topics:
            lines.append(f"Avoid these topics: {', '.join(self.avoid_topics)}")
        return "\n".join(lines)


def build_response_intent(
    action,  # PlannedAction
    state,   # ConversationState
    intent,  # SocialIntent
    mood=None,  # MoodVector
    saturated_topics: list[str] | None = None,
) -> ResponseIntent:
    """Build a ResponseIntent from the planned action and current conversation state."""
    from app.conversation.action_planner import Action

    planned_action = action.action
    effort = state.current_effort_level

    # Tone from mood
    tone = "neutral"
    if mood:
        if mood.irritation > 0.5:
            tone = "dry"
        elif mood.energy < 0.3:
            tone = "low_energy"
        elif mood.arousal > 0.7 and mood.valence > 0.0:
            tone = "playful"
        elif mood.affection > 0.5:
            tone = "warm"
        elif mood.stress > 0.6:
            tone = "slightly_stressed"

    # Language hint from effort + social mode
    if state.social_mode == "banter":
        language_hint = "hinglish"
    elif state.social_mode == "serious":
        language_hint = "mixed"
    else:
        language_hint = "hinglish"

    # Bubble count from effort + action
    if effort == "MINIMAL" or planned_action in (Action.REACT, Action.WAIT):
        bubble_count = 1
    elif effort == "HIGH" or planned_action in (Action.SUPPORT, Action.CONTINUE_STORY):
        bubble_count = min(3, 2)
    elif planned_action in (Action.TEASE, Action.PLAYFUL_COUNTER, Action.EXPLAIN):
        bubble_count = min(2, 2)
    else:
        bubble_count = 1

    return ResponseIntent(
        social_act=intent.social_act,
        response_action=str(planned_action.value if hasattr(planned_action, 'value') else planned_action),
        goal=action.conversational_goal,
        effort=effort,
        tone=tone,
        language_hint=language_hint,
        bubble_count_target=bubble_count,
        allow_question=not state.in_interview_mode,
        avoid_topics=saturated_topics or [],
    )


def format_now_block(dt: datetime | None = None) -> str:
    """Format the current IST date/time as a structured [NOW] block for the prompt."""
    ist = _get_ist_zone()
    dt = dt or datetime.now(ist)
    day = dt.strftime("%A")                    # Thursday
    date_str = dt.strftime("%d %B %Y")
    hour = dt.hour
    minute = dt.strftime("%M")

    if hour < 12:
        period = "morning"
        if hour == 0:
            time_str = f"12:{minute} AM"
        else:
            time_str = f"{hour}:{minute} AM"
    elif hour == 12:
        period = "afternoon"
        time_str = f"12:{minute} PM"
    elif hour < 17:
        period = "afternoon"
        time_str = f"{hour - 12}:{minute} PM"
    elif hour < 20:
        period = "evening"
        time_str = f"{hour - 12}:{minute} PM"
    else:
        period = "night"
        time_str = f"{hour - 12}:{minute} PM"

    # Handle midnight edge case
    if hour == 0:
        period = "night (very late)"

    return f"[NOW]\n{day}, {date_str} — {time_str} IST ({period})"

