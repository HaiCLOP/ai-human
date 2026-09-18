"""Dynamic Mood Engine — 11-dimension emotional state with per-conversation persistence and decay."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace

from app.core.logging import get_logger
from app.routine.models import AvailabilityState, MoodVector
from app.storage.repositories import AcademicExamRecord, HomeworkTaskRecord, LifeEventRecord

logger = get_logger("routine.mood")

# Per-conversation in-memory emotional state (resets on process restart — acceptable for now)
_conversation_mood: dict[str, MoodVector] = {}

# Track last activity per conversation to gate situational deltas (only apply on change)
_last_activity: dict[str, str] = {}

# Baseline mood Vesper returns to over time
_BASELINE = MoodVector(
    valence=0.2,
    arousal=0.5,
    affection=0.3,
    stress=0.3,
    energy=0.7,
    social_energy=0.6,
    confidence=0.6,
    embarrassment=0.0,
    curiosity=0.5,
    irritation=0.0,
    sadness=0.0,
    summary_note="Balanced, relatable everyday demeanor.",
)

# Decay rate per message turn — emotions move 2% toward baseline each turn (was 10% — too aggressive)
_DECAY_RATE = 0.02

# Hard floor for energy from conversation alone (sleep is unrestricted)
_ENERGY_FLOOR = 0.15

# Maximum change per single event (prevents wild swings)
_MAX_EVENT_DELTA = 0.20


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, v)), 3)


def _clamp_valence(v: float) -> float:
    return round(max(-1.0, min(1.0, v)), 3)


class MoodEngine:
    """Calculates and tracks continuous character emotional state."""

    @staticmethod
    def get_mood(conversation_id: str) -> MoodVector:
        """Retrieve current mood for a conversation, initializing to baseline if first time."""
        return _conversation_mood.get(conversation_id, copy.deepcopy(_BASELINE))

    @staticmethod
    def decay_mood(conversation_id: str) -> MoodVector:
        """Apply natural decay toward baseline — call once per conversation turn."""
        current = MoodEngine.get_mood(conversation_id)
        b = _BASELINE
        r = _DECAY_RATE

        decayed = replace(
            current,
            valence=_clamp_valence(current.valence + (b.valence - current.valence) * r),
            arousal=_clamp(current.arousal + (b.arousal - current.arousal) * r),
            affection=_clamp(current.affection + (b.affection - current.affection) * r),
            stress=_clamp(current.stress + (b.stress - current.stress) * r),
            energy=_clamp(current.energy + (b.energy - current.energy) * r),
            social_energy=_clamp(current.social_energy + (b.social_energy - current.social_energy) * r),
            confidence=_clamp(current.confidence + (b.confidence - current.confidence) * r),
            embarrassment=_clamp(current.embarrassment + (b.embarrassment - current.embarrassment) * r),
            curiosity=_clamp(current.curiosity + (b.curiosity - current.curiosity) * r),
            irritation=_clamp(current.irritation + (b.irritation - current.irritation) * r),
            sadness=_clamp(current.sadness + (b.sadness - current.sadness) * r),
        )
        _conversation_mood[conversation_id] = decayed
        return decayed

    @staticmethod
    def apply_event(conversation_id: str, event: str, magnitude: float = 0.15) -> MoodVector:
        """Apply an emotional event to the current mood state.

        Events: 'positive_interaction', 'banter', 'affection_received', 'insult_hostile',
                'ignored', 'compliment', 'sad_topic', 'embarrassing_moment', 'funny_moment'
        """
        current = MoodEngine.get_mood(conversation_id)
        m = magnitude

        deltas: dict[str, float] = {}

        if event == "positive_interaction":
            deltas = {"valence": m, "arousal": m * 0.5, "social_energy": m * 0.5}
        elif event == "banter":
            deltas = {"valence": m * 0.5, "arousal": m, "confidence": m * 0.3}
        elif event == "affection_received":
            deltas = {"affection": m, "valence": m * 0.5, "embarrassment": m * 0.2}
        elif event == "insult_hostile":
            deltas = {"irritation": m, "confidence": -m * 0.2, "valence": -m * 0.3}
        elif event == "ignored":
            deltas = {"irritation": m * 0.5, "social_energy": -m * 0.3, "sadness": m * 0.2}
        elif event == "compliment":
            deltas = {"valence": m * 0.7, "confidence": m * 0.3, "embarrassment": m * 0.15}
        elif event == "sad_topic":
            deltas = {"sadness": m, "valence": -m * 0.5, "arousal": -m * 0.2}
        elif event == "embarrassing_moment":
            deltas = {"embarrassment": m, "confidence": -m * 0.3, "arousal": m * 0.2}
        elif event == "funny_moment":
            deltas = {"valence": m, "arousal": m * 0.5, "social_energy": m * 0.3}

        updated = replace(
            current,
            valence=_clamp_valence(current.valence + deltas.get("valence", 0.0)),
            arousal=_clamp(current.arousal + deltas.get("arousal", 0.0)),
            affection=_clamp(current.affection + deltas.get("affection", 0.0)),
            stress=_clamp(current.stress + deltas.get("stress", 0.0)),
            energy=_clamp(current.energy + deltas.get("energy", 0.0)),
            social_energy=_clamp(current.social_energy + deltas.get("social_energy", 0.0)),
            confidence=_clamp(current.confidence + deltas.get("confidence", 0.0)),
            embarrassment=_clamp(current.embarrassment + deltas.get("embarrassment", 0.0)),
            curiosity=_clamp(current.curiosity + deltas.get("curiosity", 0.0)),
            irritation=_clamp(current.irritation + deltas.get("irritation", 0.0)),
            sadness=_clamp(current.sadness + deltas.get("sadness", 0.0)),
        )
        _conversation_mood[conversation_id] = updated
        return updated

    @staticmethod
    def compute_mood(
        conversation_id: str,
        current_activity: str,
        availability: AvailabilityState,
        upcoming_exams: list[AcademicExamRecord] | None = None,
        pending_homework: list[HomeworkTaskRecord] | None = None,
        recent_events: list[LifeEventRecord] | None = None,
    ) -> MoodVector:
        """Compute mood from current life state, decay current conversation state, and merge."""

        # 1. Retrieve and decay persisted conversation mood
        decayed = MoodEngine.decay_mood(conversation_id)

        # 2. Only apply situational deltas when the activity actually changes
        prev_activity = _last_activity.get(conversation_id, "")
        activity_changed = current_activity != prev_activity
        _last_activity[conversation_id] = current_activity

        energy_delta = 0.0
        stress_delta = 0.0
        arousal_delta = 0.0
        social_delta = 0.0

        if activity_changed:
            act_lower = current_activity.lower()

            if availability == AvailabilityState.AT_SCHOOL or "school" in act_lower:
                energy_delta = -0.10
                stress_delta = 0.08
                arousal_delta = 0.03
                social_delta = -0.03
            elif availability == AvailabilityState.AT_TUITION or "tuition" in act_lower:
                energy_delta = -0.15
                stress_delta = 0.12
                arousal_delta = -0.05
                social_delta = -0.08
            elif availability == AvailabilityState.STUDYING or any(
                w in act_lower for w in ("homework", "study", "revision")
            ):
                stress_delta = 0.08
                energy_delta = -0.06
                social_delta = -0.10
            elif availability == AvailabilityState.SLEEPING:
                energy_delta = -0.50
                arousal_delta = -0.40
                social_delta = -0.40
            elif any(w in act_lower for w in ("rest", "chill", "leisure")):
                energy_delta = 0.08
                stress_delta = -0.08
                arousal_delta = 0.03
                social_delta = 0.08
            elif any(w in act_lower for w in ("friends", "outing")):
                arousal_delta = 0.10
                social_delta = 0.15
                energy_delta = 0.03

        # 3. Academic pressure (applied each time regardless of activity change — exams don't go away)
        exams = upcoming_exams or []
        hw = pending_homework or []

        if any(e.importance == "HIGH" for e in exams):
            stress_delta += 0.15
            energy_delta -= 0.03
        elif exams:
            stress_delta += 0.05

        if any(h.priority == "HIGH" for h in hw):
            stress_delta += 0.05

        # 4. Apply situational deltas onto decayed state, with energy floor
        raw_energy = _clamp(decayed.energy + energy_delta)
        # Enforce energy floor (not applied during sleep)
        if availability != AvailabilityState.SLEEPING:
            raw_energy = max(raw_energy, _ENERGY_FLOOR)

        merged = replace(
            decayed,
            energy=raw_energy,
            stress=_clamp(decayed.stress + stress_delta),
            arousal=_clamp(decayed.arousal + arousal_delta),
            social_energy=_clamp(decayed.social_energy + social_delta),
        )

        # 5. Build human-readable summary note
        notes: list[str] = []
        if merged.stress >= 0.70:
            notes.append("Quite stressed right now. Brief, deadpan, low-patience responses. Don't bring up school unless asked.")
        elif merged.stress >= 0.50:
            notes.append("A bit stressed. Keep it chill and don't overthink replies.")

        if merged.energy <= 0.30:
            notes.append("Tired and low energy. Minimal effort, dry delivery.")
        elif merged.energy >= 0.75 and merged.social_energy >= 0.65:
            notes.append("High energy and sociable. Witty banter, playful teasing are natural.")

        if merged.irritation >= 0.50:
            notes.append("A little irritated. Can be slightly impatient or blunt.")

        if merged.sadness >= 0.40:
            notes.append("Feeling a bit down. Quieter, more subdued responses are natural.")

        if merged.valence >= 0.60:
            notes.append("In a good mood overall.")
        elif merged.valence <= -0.20:
            notes.append("Not in the best mood.")

        if not notes:
            notes.append("Balanced, relatable everyday demeanor.")

        final = replace(merged, summary_note=" ".join(notes))
        _conversation_mood[conversation_id] = final

        logger.info(
            "mood.computed",
            conversation_id=conversation_id,
            energy=final.energy,
            stress=final.stress,
            valence=final.valence,
            arousal=final.arousal,
            social_energy=final.social_energy,
            irritation=final.irritation,
        )
        return final

    @staticmethod
    def to_prompt_descriptor(mood: MoodVector) -> str:
        """Convert mood vector to a rich, human-readable descriptor for the prompt."""
        parts: list[str] = []

        # Energy
        if mood.energy >= 0.75:
            parts.append("energetic")
        elif mood.energy <= 0.35:
            parts.append("tired/low-energy")
        else:
            parts.append("normal energy")

        # Valence
        if mood.valence >= 0.5:
            parts.append("good mood")
        elif mood.valence <= -0.3:
            parts.append("bad mood")

        # Stress
        if mood.stress >= 0.65:
            parts.append("stressed")
        elif mood.stress <= 0.2:
            parts.append("relaxed")

        # Specific emotions
        if mood.irritation >= 0.5:
            parts.append("slightly irritated")
        if mood.embarrassment >= 0.4:
            parts.append("a bit embarrassed")
        if mood.sadness >= 0.4:
            parts.append("feeling low")
        if mood.confidence >= 0.75:
            parts.append("confident")
        elif mood.confidence <= 0.35:
            parts.append("uncertain/insecure")
        if mood.social_energy >= 0.7:
            parts.append("sociable/chatty")
        elif mood.social_energy <= 0.3:
            parts.append("wants to be left alone")

        descriptor = ", ".join(parts) if parts else "neutral"
        return f"{descriptor}. {mood.summary_note}"
