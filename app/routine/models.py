"""Data structures and state definitions for daily routine and availability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AvailabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    AWAY = "AWAY"
    SLEEPING = "SLEEPING"
    AT_SCHOOL = "AT_SCHOOL"
    AT_TUITION = "AT_TUITION"
    STUDYING = "STUDYING"
    EATING = "EATING"
    FAMILY_TIME = "FAMILY_TIME"


@dataclass
class RoutineActivity:
    activity: str
    start: str  # HH:MM format (24-hour)
    end: str    # HH:MM format
    location: str
    availability: AvailabilityState
    reason: str | None = None


@dataclass
class UpcomingCommitment:
    activity: str
    start_time: str
    end_time: str
    location: str
    minutes_until_start: int


@dataclass
class DepartureDecision:
    should_depart_now: bool
    activity: str
    start_time: str
    end_time: str
    away_until_iso: str
    departure_reason: str
    estimated_away_hours: float


@dataclass
class MoodVector:
    # Valence: -1.0 (very negative) to 1.0 (very positive) — overall emotional tone
    valence: float = 0.2
    # Arousal: 0.0 (calm/lethargic) to 1.0 (excited/energized)
    arousal: float = 0.5
    # Affection: 0.0 (indifferent) to 1.0 (warm/caring) — toward the person being talked to
    affection: float = 0.3
    # Stress: 0.0 (relaxed) to 1.0 (overwhelmed/anxious)
    stress: float = 0.3
    # Energy: 0.0 (exhausted) to 1.0 (vital/alert)
    energy: float = 0.7
    # Social energy: 0.0 (wants to be alone) to 1.0 (actively sociable)
    social_energy: float = 0.6
    # Confidence: 0.0 (insecure) to 1.0 (bold/self-assured)
    confidence: float = 0.6
    # Embarrassment: 0.0 (none) to 1.0 (deeply embarrassed)
    embarrassment: float = 0.0
    # Curiosity: 0.0 (disinterested) to 1.0 (highly curious)
    curiosity: float = 0.5
    # Irritation: 0.0 (none) to 1.0 (annoyed/irritated)
    irritation: float = 0.0
    # Sadness: 0.0 (none) to 1.0 (genuinely sad)
    sadness: float = 0.0
    summary_note: str = "Balanced, relatable everyday demeanor."


@dataclass
class AcademicContext:
    subjects_summary: list[str]
    upcoming_exams_summary: list[str]
    pending_homework_summary: list[str]


@dataclass
class VesperLifeState:
    """Authoritative simulated life state for Vesper."""

    activity: str
    location: str
    current_task: str
    energy: float
    mood_label: str
    free_time: bool
    next_event: str
    unfinished_thought: str

    def to_prompt_text(self) -> str:
        return (
            f"[VESPER LIFE STATE — AUTHORITATIVE]\n"
            f"activity: {self.activity}\n"
            f"location: {self.location}\n"
            f"current_task: {self.current_task}\n"
            f"energy: {round(self.energy, 2)}\n"
            f"mood: {self.mood_label}\n"
            f"free_time: {'yes' if self.free_time else 'no'}\n"
            f"next_event: {self.next_event}\n"
            f"unfinished_thought: {self.unfinished_thought}"
        )
