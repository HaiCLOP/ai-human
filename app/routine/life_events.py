"""Life simulation event generator and academic context coordinator."""

from __future__ import annotations

import uuid
from typing import Any

from app.core.logging import get_logger
from app.routine.models import AcademicContext
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import AcademicRepository, LifeEventRepository

logger = get_logger("routine.life_events")


class LifeEventManager:
    """Manages fictional academic curriculum, homework tasks, and daily life events."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.academic_repo = AcademicRepository(self.db)
        self.life_repo = LifeEventRepository(self.db)

    def sync_academics_from_config(self, academics_config: dict[str, Any]) -> None:
        """Seed or sync subjects, exams, and homework from character.yaml."""
        subjects = academics_config.get("subjects", {})
        for sub_id, data in subjects.items():
            sub_name = sub_id.capitalize()
            curr_topic = data.get("current_topic", "General")
            self.academic_repo.upsert_subject(sub_id, sub_name, curr_topic)

            exam = data.get("upcoming_exam")
            if exam:
                self.academic_repo.add_exam(
                    exam_id=f"exam_{sub_id}_{exam.get('date', 'soon')}",
                    subject_id=sub_id,
                    topic=exam.get("topic", curr_topic),
                    exam_date=exam.get("date", "2026-09-20"),
                    importance=exam.get("importance", "HIGH").upper(),
                )

        homework_list = academics_config.get("homework", [])
        for hw in homework_list:
            self.academic_repo.upsert_homework(
                task_id=hw.get("id", str(uuid.uuid4())),
                subject_id=hw.get("subject", "general"),
                description=hw.get("description", ""),
                due_date=hw.get("due_date", "2026-09-20"),
                priority=hw.get("priority", "MEDIUM").upper(),
            )

    def get_academic_context(self) -> AcademicContext:
        """Fetch human-readable summaries of subjects, exams, and pending homework."""
        exams = self.academic_repo.get_upcoming_exams()
        homework = self.academic_repo.get_pending_homework()

        exam_summaries = [
            f"{e.subject_id.capitalize()}: {e.topic} (Date: {e.exam_date}, Priority: {e.importance})"
            for e in exams
        ]
        hw_summaries = [
            f"[{h.subject_id.capitalize()}] {h.description} (Due: {h.due_date})"
            for h in homework
        ]

        return AcademicContext(
            subjects_summary=["Mathematics (Quadratic Equations)", "Physics (Laws of Motion)", "Chemistry (Chemical Bonding)"],
            upcoming_exams_summary=exam_summaries,
            pending_homework_summary=hw_summaries,
        )

    def get_unmentioned_events_for_prompt(self) -> list[str]:
        """Fetch recent life events not yet shared in chat."""
        events = self.life_repo.get_recent_unmentioned_events(limit=2)
        notes: list[str] = []
        for ev in events:
            notes.append(f"Recent event today: {ev.headline}")
            self.life_repo.mark_event_mentioned(ev.event_id)
        return notes
