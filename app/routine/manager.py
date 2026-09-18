"""Routine and availability manager resolving real-time schedule slots and departures."""

from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.routine.models import (
    AvailabilityState,
    DepartureDecision,
    RoutineActivity,
    UpcomingCommitment,
)
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import (
    AvailabilityRepository,
    RoutineRepository,
    RoutineSlotRecord,
)

logger = get_logger("routine.manager")


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
        now = current_dt or datetime.now(timezone.utc)
        avail_record = self.avail_repo.get_or_create(character_id)

        # 1. Enforce active away_until override if not yet expired
        if avail_record.away_until:
            try:
                away_dt = datetime.fromisoformat(avail_record.away_until.replace("Z", "+00:00"))
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
            if end_t == dt_time(23, 59):
                is_match = (start_t <= now_time <= end_t)
            else:
                is_match = (start_t <= now_time < end_t)

            if is_match:
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

        # Default fallback
        return AvailabilityState.AVAILABLE, RoutineActivity(
            activity="free_time",
            start="00:00",
            end="23:59",
            location="home",
            availability=AvailabilityState.AVAILABLE,
        )

    def check_upcoming_commitment(
        self,
        current_dt: datetime | None = None,
        character_id: str = "default",
    ) -> UpcomingCommitment | None:
        """Check if an unavailable commitment is approaching within the warning window."""
        now = current_dt or datetime.now(timezone.utc)
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
        now = current_dt or datetime.now(timezone.utc)
        state, activity = self.resolve_availability(now, character_id=character_id)

        # If already marked away in database, no need to trigger departure again
        avail_record = self.avail_repo.get_or_create(character_id)
        if avail_record.away_until:
            return None

        # If current activity is an unavailable state (tuition, school, sleep, etc.)
        if state != AvailabilityState.AVAILABLE:
            end_t = _parse_hh_mm(activity.end) if activity.end else dt_time(23, 59)
            # Construct away_until datetime today
            away_until_dt = datetime.combine(now.date(), end_t, tzinfo=timezone.utc)
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
