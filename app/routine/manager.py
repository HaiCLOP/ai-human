"""Routine and availability manager resolving real-time schedule slots and departures."""

from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.routine.models import (
    AvailabilityState,
    DepartureDecision,
    MoodVector,
    RoutineActivity,
    UpcomingCommitment,
    VesperLifeState,
)
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import (
    AvailabilityRepository,
    RoutineRepository,
    RoutineSlotRecord,
)

logger = get_logger("routine.manager")


IST = timezone(timedelta(hours=5, minutes=30))


def to_ist(dt: datetime | None = None) -> datetime:
    """Convert datetime to Indian Standard Time (Asia/Kolkata, UTC+5:30).

    When dt is None, returns current real-time in IST.
    When dt is provided (e.g. simulated time or test datetime), preserves
    its clock time for schedule resolution.
    """
    if dt is None:
        return datetime.now(IST)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt


def _parse_hh_mm(time_str: str) -> dt_time:
    parts = time_str.split(":")
    return dt_time(hour=int(parts[0]), minute=int(parts[1]))


class RoutineManager:
    """Manages weekly schedule templates, real-time availability states, and departure triggers."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        routine_config: dict[str, Any] | None = None,
        departure_warning_minutes: int = 15,
    ):
        self.db = db or get_db_manager()
        self.routine_repo = RoutineRepository(self.db)
        self.avail_repo = AvailabilityRepository(self.db)
        self.warning_window_minutes = departure_warning_minutes

        if routine_config:
            self.sync_routine_templates(routine_config)

    def sync_routine_templates(self, routine_config: dict[str, Any]) -> None:
        """Populate routine_templates table from YAML weekly schedule."""
        slots: list[RoutineSlotRecord] = []
        for day, activities in routine_config.items():
            for idx, act in enumerate(activities):
                slots.append(
                    RoutineSlotRecord(
                        template_id=f"{day.lower()}_{idx}_{act['activity']}",
                        day_of_week=day.lower(),
                        start_time=act["start"],
                        end_time=act["end"],
                        activity=act["activity"],
                        location=act.get("location", "home"),
                        default_availability=act.get("availability", "AVAILABLE"),
                    )
                )
        self.routine_repo.set_routine_slots(slots)
        logger.info("routine.templates_synced", total_slots=len(slots))

    def resolve_availability(
        self,
        current_dt: datetime | None = None,
        character_id: str = "default",
    ) -> tuple[AvailabilityState, RoutineActivity]:
        """Deterministically compute current availability state and active commitment."""
        now = to_ist(current_dt)
        avail_record = self.avail_repo.get_or_create(character_id)

        # 1. Enforce active away_until override if not yet expired
        if avail_record.away_until:
            try:
                away_dt = datetime.fromisoformat(avail_record.away_until.replace("Z", "+00:00"))
                if away_dt.tzinfo is None:
                    away_dt = away_dt.replace(tzinfo=IST)
                else:
                    away_dt = away_dt.astimezone(IST)
                if now < away_dt:
                    state = AvailabilityState(avail_record.current_state)
                    activity = RoutineActivity(
                        activity=avail_record.current_activity,
                        start="",
                        end=away_dt.strftime("%H:%M"),
                        location=avail_record.current_location,
                        availability=state,
                        reason="Active away commitment",
                    )
                    return state, activity
                else:
                    # Away window expired -> clear away_until
                    self.avail_repo.update_state(
                        character_id=character_id,
                        current_state="AVAILABLE",
                        current_activity="free_time",
                        current_location="home",
                        away_until=None,
                        warning_sent_for_activity=None,
                    )
            except Exception as e:
                logger.warning("routine.failed_parsing_away_until", error=str(e))

        # 2. Query canonical weekly schedule for current day and time
        day_name = now.strftime("%A").lower()
        now_time = now.time()
        slots = self.routine_repo.get_slots_for_day(day_name)

        matched_slot: RoutineSlotRecord | None = None
        for s in slots:
            start_t = _parse_hh_mm(s.start_time)
            end_t = _parse_hh_mm(s.end_time)

            if start_t <= end_t:
                if start_t <= now_time < end_t:
                    matched_slot = s
                    break
            else:
                # Slot crosses midnight (e.g. sleep 23:00 -> 07:00)
                if now_time >= start_t or now_time < end_t:
                    matched_slot = s
                    break

        if matched_slot:
            state = AvailabilityState(matched_slot.default_availability)
            activity = RoutineActivity(
                activity=matched_slot.activity,
                start=matched_slot.start_time,
                end=matched_slot.end_time,
                location=matched_slot.location,
                availability=state,
            )
            return state, activity

        # Fallback default: available free time at home
        fallback_activity = RoutineActivity(
            activity="free_time",
            start="00:00",
            end="23:59",
            location="home",
            availability=AvailabilityState.AVAILABLE,
        )
        return AvailabilityState.AVAILABLE, fallback_activity

    def check_upcoming_commitment(
        self,
        current_dt: datetime | None = None,
        character_id: str = "default",
    ) -> UpcomingCommitment | None:
        """Check if an unavailable commitment is approaching within the warning window."""
        now = to_ist(current_dt)
        day_name = now.strftime("%A").lower()
        now_time = now.time()
        slots = self.routine_repo.get_slots_for_day(day_name)

        for s in slots:
            # Only track commitments that make her unavailable
            if s.default_availability == "AVAILABLE":
                continue

            start_t = _parse_hh_mm(s.start_time)
            # Calculate delta minutes today
            now_mins = now_time.hour * 60 + now_time.minute
            start_mins = start_t.hour * 60 + start_t.minute
            delta_mins = start_mins - now_mins

            if 0 < delta_mins <= self.warning_window_minutes:
                # Check if warning was already sent for this activity
                avail_record = self.avail_repo.get_or_create(character_id)
                if avail_record.warning_sent_for_activity == s.template_id:
                    return None  # Already warned

                return UpcomingCommitment(
                    activity=s.activity,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    location=s.location,
                    minutes_until_start=delta_mins,
                )

        return None

    def evaluate_departure(
        self,
        current_dt: datetime | None = None,
        character_id: str = "default",
    ) -> DepartureDecision | None:
        """Check if an unavailable commitment has arrived, requiring departure from conversation."""
        now = to_ist(current_dt)
        state, activity = self.resolve_availability(now, character_id=character_id)

        # If already marked away in database, no need to trigger departure again
        avail_record = self.avail_repo.get_or_create(character_id)
        if avail_record.away_until:
            return None

        # If current activity is an unavailable state (tuition, school, sleep, etc.)
        if state != AvailabilityState.AVAILABLE:
            end_t = _parse_hh_mm(activity.end) if activity.end else dt_time(23, 59)
            # Construct away_until datetime today
            away_until_dt = datetime.combine(now.date(), end_t, tzinfo=now.tzinfo)
            if away_until_dt <= now:
                away_until_dt += timedelta(days=1)

            diff_hours = round((away_until_dt - now).total_seconds() / 3600.0, 1)

            return DepartureDecision(
                should_depart_now=True,
                activity=activity.activity,
                start_time=activity.start,
                end_time=activity.end,
                away_until_iso=away_until_dt.isoformat(),
                departure_reason=f"Scheduled {activity.activity} at {activity.location}",
                estimated_away_hours=diff_hours,
            )

        return None

    def mark_departure(
        self,
        character_id: str,
        activity: str,
        new_state: AvailabilityState,
        location: str,
        away_until_iso: str,
    ) -> None:
        """Update availability state to away."""
        self.avail_repo.update_state(
            character_id=character_id,
            current_state=new_state.value,
            current_activity=activity,
            current_location=location,
            away_until=away_until_iso,
            warning_sent_for_activity=None,
        )
        logger.info(
            "routine.departure_marked",
            character_id=character_id,
            activity=activity,
            away_until=away_until_iso,
        )

    def mark_warning_sent(self, character_id: str, template_id: str) -> None:
        """Record that warning was sent to prevent duplicate announcements."""
        rec = self.avail_repo.get_or_create(character_id)
        self.avail_repo.update_state(
            character_id=character_id,
            current_state=rec.current_state,
            current_activity=rec.current_activity,
            current_location=rec.current_location,
            away_until=rec.away_until,
            warning_sent_for_activity=template_id,
        )

    def build_life_state(
        self,
        current_dt: datetime | None = None,
        now: datetime | None = None,
        character_id: str = "default",
        pending_homework: list[Any] | None = None,
        upcoming_exams: list[Any] | None = None,
        mood: MoodVector | None = None,
        **kwargs: Any,
    ) -> VesperLifeState:
        """Construct authoritative VesperLifeState for the current conversational moment."""
        target_dt = current_dt or now or kwargs.get("simulated_time")
        now_dt = to_ist(target_dt)
        state, activity = self.resolve_availability(now_dt, character_id=character_id)

        now = now_dt
        # Find next event
        day_name = now.strftime("%A").lower()
        now_time = now.time()
        slots = self.routine_repo.get_slots_for_day(day_name)
        next_event = "rest / sleep"
        for s in slots:
            start_t = _parse_hh_mm(s.start_time)
            if start_t > now_time:
                next_event = f"{s.activity} at {s.start_time}"
                break

        # Determine specific task and thought
        hw_desc = pending_homework[0].description if pending_homework else ""
        act_name = activity.activity.lower()

        if "school" in act_name:
            task = "school periods and classes"
            thought = "waiting for the final bell"
        elif "tuition" in act_name:
            task = "maths coaching with Sharma Sir"
            thought = "trying to understand quadratic steps"
        elif "homework" in act_name or "study" in act_name:
            task = hw_desc or "physics numerical problems on friction"
            thought = "physics question 7 incline plane calculation"
        elif "snack" in act_name or "rest" in act_name:
            task = "eating evening snack and relaxing at home"
            thought = "thinking about what to listen to on Spotify"
        elif "dinner" in act_name:
            task = "having dinner with family"
            thought = "listening to mom talking about board exams"
        elif "sleep" in act_name:
            task = "trying to fall asleep"
            thought = "half asleep in bed"
        else:
            task = "relaxing on phone"
            thought = "scrolling Instagram reels"

        energy = mood.energy if mood else 0.6
        stress = mood.stress if mood else 0.3

        if energy < 0.4:
            mood_label = "tired / drained"
        elif stress > 0.6:
            mood_label = "stressed about studies"
        elif mood and mood.irritation > 0.4:
            mood_label = "a bit annoyed"
        elif energy > 0.7:
            mood_label = "chill and energized"
        else:
            mood_label = "casual / chill"

        is_free = (
            state == AvailabilityState.AVAILABLE
            and act_name in ("free_time", "free_time_casual", "rest", "rest_and_snacks")
        )

        return VesperLifeState(
            activity=activity.activity,
            location=activity.location,
            current_task=task,
            energy=energy,
            mood_label=mood_label,
            free_time=is_free,
            next_event=next_event,
            unfinished_thought=thought,
        )
