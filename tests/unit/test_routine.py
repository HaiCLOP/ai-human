"""Unit tests for RoutineManager, MoodEngine, and LifeEventManager."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from app.ai.persona import load_character_profile
from app.routine.life_events import LifeEventManager
from app.routine.manager import RoutineManager
from app.routine.models import AvailabilityState
from app.routine.mood import MoodEngine
from app.storage.database import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_routine.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


def test_routine_schedule_resolution(temp_db):
    profile = load_character_profile()
    mgr = RoutineManager(db=temp_db, routine_config=profile.routine)

    # 1. Monday at 10:00 -> School
    mon_school = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc)  # Sept 21, 2026 is Monday
    state, act = mgr.resolve_availability(mon_school)
    assert state == AvailabilityState.AT_SCHOOL
    assert act.activity == "school"

    # 2. Monday at 15:30 -> Rest / Available
    mon_rest = datetime(2026, 9, 21, 15, 30, tzinfo=timezone.utc)
    state, act = mgr.resolve_availability(mon_rest)
    assert state == AvailabilityState.AVAILABLE
    assert act.activity == "rest"

    # 3. Monday at 17:30 -> Tuition
    mon_tuition = datetime(2026, 9, 21, 17, 30, tzinfo=timezone.utc)
    state, act = mgr.resolve_availability(mon_tuition)
    assert state == AvailabilityState.AT_TUITION
    assert act.activity == "maths_tuition"

    # 4. Monday at 23:45 -> Sleeping
    mon_sleep = datetime(2026, 9, 21, 23, 45, tzinfo=timezone.utc)
    state, act = mgr.resolve_availability(mon_sleep)
    assert state == AvailabilityState.SLEEPING
    assert act.activity == "sleep"


def test_upcoming_commitment_warning(temp_db):
    profile = load_character_profile()
    mgr = RoutineManager(db=temp_db, routine_config=profile.routine, departure_warning_minutes=15)

    # Monday 16:50 -> 10 minutes before 17:00 tuition
    mon_pre_tuition = datetime(2026, 9, 21, 16, 50, tzinfo=timezone.utc)
    upcoming = mgr.check_upcoming_commitment(mon_pre_tuition)
    assert upcoming is not None
    assert upcoming.activity == "maths_tuition"
    assert upcoming.minutes_until_start == 10

    # Monday 16:30 -> 30 minutes before -> Outside 15 min window
    mon_far = datetime(2026, 9, 21, 16, 30, tzinfo=timezone.utc)
    upcoming_far = mgr.check_upcoming_commitment(mon_far)
    assert upcoming_far is None


def test_departure_evaluation_and_away_until(temp_db):
    profile = load_character_profile()
    mgr = RoutineManager(db=temp_db, routine_config=profile.routine)

    # Monday 17:00 -> Tuition starts
    mon_1700 = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)
    departure = mgr.evaluate_departure(mon_1700)
    assert departure is not None
    assert departure.should_depart_now is True
    assert departure.activity == "maths_tuition"
    assert departure.estimated_away_hours == 2.0

    # Mark departure
    mgr.mark_departure("default", departure.activity, AvailabilityState.AT_TUITION, "tuition", departure.away_until_iso)

    # At 17:30 -> Character is away at tuition
    mon_1730 = datetime(2026, 9, 21, 17, 30, tzinfo=timezone.utc)
    state_during, act_during = mgr.resolve_availability(mon_1730)
    assert state_during == AvailabilityState.AT_TUITION

    # At 19:05 -> Away window expired -> Automatically becomes AVAILABLE
    mon_1905 = datetime(2026, 9, 21, 19, 5, tzinfo=timezone.utc)
    state_after, act_after = mgr.resolve_availability(mon_1905)
    assert state_after == AvailabilityState.AVAILABLE


def test_mood_engine_modulation():
    # Exhausting activity — MoodEngine now requires conversation_id as first arg
    mood_tuition = MoodEngine.compute_mood(
        "test_conv_tuition", "maths_tuition", AvailabilityState.AT_TUITION
    )
    assert mood_tuition.energy <= 0.65  # energy drops at tuition
    assert mood_tuition.stress >= 0.38  # stress rises at tuition (baseline 0.30 + 0.12 delta = 0.42)

    # Relaxed activity
    mood_rest = MoodEngine.compute_mood(
        "test_conv_rest", "rest", AvailabilityState.AVAILABLE
    )
    assert mood_rest.energy >= 0.60   # energy at rest (delta-gated, only applies on first activity change)
    assert mood_rest.social_energy >= 0.50  # social energy up when resting



def test_life_events_and_academics(temp_db):
    profile = load_character_profile()
    life_mgr = LifeEventManager(temp_db)
    life_mgr.sync_academics_from_config(profile.academics)

    context = life_mgr.get_academic_context()
    assert len(context.upcoming_exams_summary) >= 1
    assert any("Mathematics" in s for s in context.upcoming_exams_summary)
    assert len(context.pending_homework_summary) >= 1
