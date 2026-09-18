"""Typed repositories for SQLite entities: conversations, messages, memories, style, and audit."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.storage.database import DatabaseManager, get_db_manager


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationRecord:
    conversation_id: str
    platform: str
    participant_handle: str
    participant_name: str | None
    status: str
    created_at: str
    updated_at: str


@dataclass
class MessageRecord:
    message_id: str
    conversation_id: str
    fingerprint: str
    sender_type: str  # 'USER' or 'CHARACTER'
    sender_handle: str
    content: str
    timestamp_utc: str
    status: str  # 'RECEIVED', 'PROCESSING', 'SENT', 'FAILED', 'IGNORED'
    retry_count: int
    processed_at: str | None


@dataclass
class MemoryRecord:
    memory_id: str
    conversation_id: str
    memory_type: str  # 'FACT', 'PREFERENCE', 'RUNNING_JOKE', 'TOPIC'
    statement: str
    confidence: float
    access_count: int
    created_at: str
    last_accessed_at: str


@dataclass
class RelationshipRecord:
    conversation_id: str
    familiarity: float
    playfulness: float
    sarcasm_tolerance: float
    trust: float
    running_jokes_json: str
    updated_at: str


@dataclass
class StyleProfileRecord:
    profile_id: str
    conversation_id: str
    avg_sentence_length: float
    formality_score: float
    emoji_density: float
    slang_vocabulary_json: str
    updated_at: str


@dataclass
class StyleObservationRecord:
    observation_id: str
    profile_id: str
    trait_name: str
    trait_value: str
    confidence: float
    evidence_count: int
    first_seen: str
    last_seen: str


@dataclass
class DocumentChunkRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: bytes
    metadata_json: str


class ConversationRepository:
    """Manages conversations and participant records."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def get_or_create(
        self,
        conversation_id: str,
        participant_handle: str,
        participant_name: str | None = None,
        platform: str = "instagram",
    ) -> ConversationRecord:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row:
                return ConversationRecord(**dict(row))

            now = _utc_now_iso()
            cursor.execute(
                """
                INSERT INTO conversations (
                    conversation_id, platform, participant_handle, participant_name, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                """,
                (conversation_id, platform, participant_handle, participant_name, now, now),
            )
            return ConversationRecord(
                conversation_id=conversation_id,
                platform=platform,
                participant_handle=participant_handle,
                participant_name=participant_name,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        finally:
            conn.close()

    def get(self, conversation_id: str) -> ConversationRecord | None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            return ConversationRecord(**dict(row)) if row else None
        finally:
            conn.close()


class MessageRepository:
    """Manages message logging, idempotency checks, and conversation history."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def exists_by_fingerprint(self, fingerprint: str) -> bool:
        """Check if message fingerprint has already been registered."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM messages WHERE fingerprint = ?",
                (fingerprint,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def is_already_handled(self, fingerprint: str) -> bool:
        """Check if message was already successfully answered or intentionally dismissed."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM messages WHERE fingerprint = ?",
                (fingerprint,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            return row[0] in ("SENT", "IGNORED")
        finally:
            conn.close()

    def record_incoming_message(
        self,
        conversation_id: str,
        fingerprint: str,
        sender_handle: str,
        content: str,
    ) -> MessageRecord:
        """Atomically insert or update a retryable incoming user message."""
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_id, retry_count FROM messages WHERE fingerprint = ?",
                (fingerprint,),
            )
            existing = cursor.fetchone()
            if existing:
                msg_id, retries = existing[0], existing[1] or 0
                cursor.execute(
                    "UPDATE messages SET status = 'PROCESSING', retry_count = ?, processed_at = ? WHERE message_id = ?",
                    (retries + 1, now, msg_id),
                )
                conn.commit()
                return MessageRecord(
                    message_id=msg_id,
                    conversation_id=conversation_id,
                    fingerprint=fingerprint,
                    sender_type="USER",
                    sender_handle=sender_handle,
                    content=content,
                    timestamp_utc=now,
                    status="PROCESSING",
                    retry_count=retries + 1,
                    processed_at=now,
                )

            message_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, fingerprint, sender_type,
                    sender_handle, content, timestamp_utc, status, retry_count
                ) VALUES (?, ?, ?, 'USER', ?, ?, ?, 'RECEIVED', 0)
                """,
                (message_id, conversation_id, fingerprint, sender_handle, content, now),
            )
            conn.commit()
            return MessageRecord(
                message_id=message_id,
                conversation_id=conversation_id,
                fingerprint=fingerprint,
                sender_type="USER",
                sender_handle=sender_handle,
                content=content,
                timestamp_utc=now,
                status="RECEIVED",
                retry_count=0,
                processed_at=None,
            )
        finally:
            conn.close()

    def record_character_message(
        self,
        conversation_id: str,
        content: str,
        character_handle: str = "CHARACTER",
    ) -> MessageRecord:
        """Record a sent character reply in 'SENT' state."""
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            message_id = str(uuid.uuid4())
            # Synthesize deterministic fingerprint for outbound message
            fingerprint = f"out_{message_id}"
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, fingerprint, sender_type,
                    sender_handle, content, timestamp_utc, status, retry_count, processed_at
                ) VALUES (?, ?, ?, 'CHARACTER', ?, ?, ?, 'SENT', 0, ?)
                """,
                (message_id, conversation_id, fingerprint, character_handle, content, now, now),
            )
            return MessageRecord(
                message_id=message_id,
                conversation_id=conversation_id,
                fingerprint=fingerprint,
                sender_type="CHARACTER",
                sender_handle=character_handle,
                content=content,
                timestamp_utc=now,
                status="SENT",
                retry_count=0,
                processed_at=now,
            )
        finally:
            conn.close()

    def update_status(self, message_id: str, status: str) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE messages SET status = ?, processed_at = ? WHERE message_id = ?",
                (status, now, message_id),
            )
        finally:
            conn.close()

    def get_recent_messages(self, conversation_id: str, limit: int = 10) -> list[MessageRecord]:
        """Fetch sliding window of recent conversation turns in chronological order."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ? AND status IN ('RECEIVED', 'SENT', 'PROCESSING')
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = cursor.fetchall()
            records = [MessageRecord(**dict(row)) for row in rows]
            records.reverse()  # Return in chronological order
            return records
        finally:
            conn.close()


class MemoryRepository:
    """Manages long-term facts, preferences, and callbacks."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def add_memory(
        self,
        conversation_id: str,
        memory_type: str,
        statement: str,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            memory_id = str(uuid.uuid4())
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (
                    memory_id, conversation_id, memory_type, statement,
                    confidence, access_count, created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (memory_id, conversation_id, memory_type, statement, confidence, now, now),
            )
            return MemoryRecord(
                memory_id=memory_id,
                conversation_id=conversation_id,
                memory_type=memory_type,
                statement=statement,
                confidence=confidence,
                access_count=0,
                created_at=now,
                last_accessed_at=now,
            )
        finally:
            conn.close()

    def get_memories_for_conversation(self, conversation_id: str) -> list[MemoryRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE conversation_id = ? AND confidence > 0.0 ORDER BY confidence DESC",
                (conversation_id,),
            )
            return [MemoryRecord(**dict(row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def touch_memory(self, memory_id: str) -> None:
        """Increment access count and update last_accessed_at."""
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE memory_id = ?
                """,
                (now, memory_id),
            )
        finally:
            conn.close()


class RelationshipRepository:
    """Tracks conversation chemistry and running jokes."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def get_or_create(self, conversation_id: str) -> RelationshipRecord:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM relationship_states WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row:
                return RelationshipRecord(**dict(row))

            now = _utc_now_iso()
            cursor.execute(
                """
                INSERT INTO relationship_states (
                    conversation_id, familiarity, playfulness, sarcasm_tolerance, trust, running_jokes_json, updated_at
                ) VALUES (?, 0.1, 0.5, 0.3, 0.2, '[]', ?)
                """,
                (conversation_id, now),
            )
            return RelationshipRecord(
                conversation_id=conversation_id,
                familiarity=0.1,
                playfulness=0.5,
                sarcasm_tolerance=0.3,
                trust=0.2,
                running_jokes_json="[]",
                updated_at=now,
            )
        finally:
            conn.close()

    def update_chemistry(
        self,
        conversation_id: str,
        familiarity: float,
        playfulness: float,
        sarcasm_tolerance: float,
        trust: float,
        running_jokes_json: str | None = None,
    ) -> None:
        self.get_or_create(conversation_id)
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            if running_jokes_json is not None:
                cursor.execute(
                    """
                    UPDATE relationship_states
                    SET familiarity = ?, playfulness = ?, sarcasm_tolerance = ?, trust = ?, running_jokes_json = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (familiarity, playfulness, sarcasm_tolerance, trust, running_jokes_json, now, conversation_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE relationship_states
                    SET familiarity = ?, playfulness = ?, sarcasm_tolerance = ?, trust = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (familiarity, playfulness, sarcasm_tolerance, trust, now, conversation_id),
                )
        finally:
            conn.close()


class StyleRepository:
    """Manages stylometric profiles and observations."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def get_or_create_profile(self, conversation_id: str) -> StyleProfileRecord:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM style_profiles WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row:
                return StyleProfileRecord(**dict(row))

            now = _utc_now_iso()
            profile_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO style_profiles (
                    profile_id, conversation_id, avg_sentence_length, formality_score, emoji_density, slang_vocabulary_json, updated_at
                ) VALUES (?, ?, 8.0, 0.2, 0.1, '{}', ?)
                """,
                (profile_id, conversation_id, now),
            )
            return StyleProfileRecord(
                profile_id=profile_id,
                conversation_id=conversation_id,
                avg_sentence_length=8.0,
                formality_score=0.2,
                emoji_density=0.1,
                slang_vocabulary_json="{}",
                updated_at=now,
            )
        finally:
            conn.close()

    def update_profile(
        self,
        profile_id: str,
        avg_sentence_length: float,
        formality_score: float,
        emoji_density: float,
        slang_vocabulary_json: str,
    ) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE style_profiles
                SET avg_sentence_length = ?, formality_score = ?, emoji_density = ?, slang_vocabulary_json = ?, updated_at = ?
                WHERE profile_id = ?
                """,
                (avg_sentence_length, formality_score, emoji_density, slang_vocabulary_json, now, profile_id),
            )
        finally:
            conn.close()

    def record_observation(
        self,
        profile_id: str,
        trait_name: str,
        trait_value: str,
        confidence: float,
    ) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM style_observations WHERE profile_id = ? AND trait_name = ?",
                (profile_id, trait_name),
            )
            row = cursor.fetchone()
            if row:
                obs_id = row["observation_id"]
                evidence_count = row["evidence_count"] + 1
                cursor.execute(
                    """
                    UPDATE style_observations
                    SET trait_value = ?, confidence = ?, evidence_count = ?, last_seen = ?
                    WHERE observation_id = ?
                    """,
                    (trait_value, confidence, evidence_count, now, obs_id),
                )
            else:
                obs_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO style_observations (
                        observation_id, profile_id, trait_name, trait_value, confidence, evidence_count, first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (obs_id, profile_id, trait_name, trait_value, confidence, now, now),
                )
        finally:
            conn.close()

    def get_observations(self, profile_id: str) -> list[StyleObservationRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM style_observations WHERE profile_id = ?",
                (profile_id,),
            )
            return [StyleObservationRecord(**dict(row)) for row in cursor.fetchall()]
        finally:
            conn.close()


class RAGRepository:
    """Manages document metadata and vectorized chunks."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def insert_document(self, document_id: str, title: str, category: str, source_path: str) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (document_id, title, category, source_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, title, category, source_path, now, now),
            )
        finally:
            conn.close()

    def insert_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        content: str,
        embedding: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO document_chunks (chunk_id, document_id, chunk_index, content, embedding, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, chunk_index, content, embedding, json.dumps(metadata or {})),
            )
        finally:
            conn.close()

    def get_all_chunks(self) -> list[DocumentChunkRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_chunks ORDER BY document_id, chunk_index")
            return [DocumentChunkRecord(**dict(row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def clear_document_chunks(self, document_id: str) -> None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        finally:
            conn.close()


class AuditRepository:
    """Records security halts, guardrail rejections, and system faults."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def record_event(
        self,
        event_type: str,
        severity: str,
        component: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> str:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            event_id = str(uuid.uuid4())
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_events (event_id, correlation_id, event_type, severity, component, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, correlation_id, event_type, severity, component, json.dumps(payload or {}), now),
            )
            return event_id
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


@dataclass
class RoutineSlotRecord:
    template_id: str
    day_of_week: str
    start_time: str
    end_time: str
    activity: str
    location: str
    default_availability: str


@dataclass
class AvailabilityStateRecord:
    character_id: str
    current_state: str
    current_activity: str
    current_location: str
    away_until: str | None
    warning_sent_for_activity: str | None
    updated_at: str


@dataclass
class AcademicExamRecord:
    exam_id: str
    subject_id: str
    topic: str
    exam_date: str
    importance: str
    status: str
    result_notes: str | None


@dataclass
class HomeworkTaskRecord:
    task_id: str
    subject_id: str
    description: str
    due_date: str
    status: str
    priority: str
    created_at: str


@dataclass
class LifeEventRecord:
    event_id: str
    category: str
    headline: str
    sentiment: str
    emotional_impact_json: str
    occurred_at: str
    mentioned_in_chat: int


class RoutineRepository:
    """Manages recurring weekly routine templates and special events."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def set_routine_slots(self, slots: list[RoutineSlotRecord]) -> None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM routine_templates")
            for s in slots:
                cursor.execute(
                    """
                    INSERT INTO routine_templates (
                        template_id, day_of_week, start_time, end_time, activity, location, default_availability
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (s.template_id, s.day_of_week, s.start_time, s.end_time, s.activity, s.location, s.default_availability),
                )
        finally:
            conn.close()

    def get_slots_for_day(self, day_of_week: str) -> list[RoutineSlotRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM routine_templates WHERE day_of_week = ? ORDER BY start_time ASC",
                (day_of_week.lower(),),
            )
            return [RoutineSlotRecord(**dict(r)) for r in cursor.fetchall()]
        finally:
            conn.close()


class AvailabilityRepository:
    """Manages real-time availability state, away_until, and departure warning flags."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def get_or_create(self, character_id: str = "default") -> AvailabilityStateRecord:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM availability_state WHERE character_id = ?", (character_id,))
            row = cursor.fetchone()
            if row:
                return AvailabilityStateRecord(**dict(row))

            now = _utc_now_iso()
            cursor.execute(
                """
                INSERT INTO availability_state (
                    character_id, current_state, current_activity, current_location, away_until, warning_sent_for_activity, updated_at
                ) VALUES (?, 'AVAILABLE', 'free_time', 'home', NULL, NULL, ?)
                """,
                (character_id, now),
            )
            return AvailabilityStateRecord(
                character_id=character_id,
                current_state="AVAILABLE",
                current_activity="free_time",
                current_location="home",
                away_until=None,
                warning_sent_for_activity=None,
                updated_at=now,
            )
        finally:
            conn.close()

    def update_state(
        self,
        character_id: str,
        current_state: str,
        current_activity: str,
        current_location: str,
        away_until: str | None = None,
        warning_sent_for_activity: str | None = None,
    ) -> None:
        self.get_or_create(character_id)
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE availability_state
                SET current_state = ?, current_activity = ?, current_location = ?, away_until = ?, warning_sent_for_activity = ?, updated_at = ?
                WHERE character_id = ?
                """,
                (current_state, current_activity, current_location, away_until, warning_sent_for_activity, now, character_id),
            )
        finally:
            conn.close()


class AcademicRepository:
    """Manages academic subjects, exams, and homework tasks."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def upsert_subject(self, subject_id: str, subject_name: str, current_topic: str) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO academic_subjects (subject_id, subject_name, current_topic, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject_id) DO UPDATE SET
                    subject_name = excluded.subject_name,
                    current_topic = excluded.current_topic,
                    updated_at = excluded.updated_at
                """,
                (subject_id, subject_name, current_topic, now),
            )
        finally:
            conn.close()

    def add_exam(self, exam_id: str, subject_id: str, topic: str, exam_date: str, importance: str = "HIGH") -> None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO academic_exams (exam_id, subject_id, topic, exam_date, importance, status)
                VALUES (?, ?, ?, ?, ?, 'SCHEDULED')
                """,
                (exam_id, subject_id, topic, exam_date, importance),
            )
        finally:
            conn.close()

    def get_upcoming_exams(self) -> list[AcademicExamRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM academic_exams WHERE status = 'SCHEDULED' ORDER BY exam_date ASC")
            return [AcademicExamRecord(**dict(r)) for r in cursor.fetchall()]
        finally:
            conn.close()

    def upsert_homework(self, task_id: str, subject_id: str, description: str, due_date: str, priority: str = "MEDIUM") -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO homework_tasks (task_id, subject_id, description, due_date, status, priority, created_at)
                VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (task_id, subject_id, description, due_date, priority, now),
            )
        finally:
            conn.close()

    def get_pending_homework(self) -> list[HomeworkTaskRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM homework_tasks WHERE status != 'DONE' ORDER BY due_date ASC")
            return [HomeworkTaskRecord(**dict(r)) for r in cursor.fetchall()]
        finally:
            conn.close()


class LifeEventRepository:
    """Records spontaneous fictional daily events."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def record_event(
        self,
        event_id: str,
        category: str,
        headline: str,
        sentiment: str = "NEUTRAL",
        emotional_impact: dict[str, float] | None = None,
    ) -> None:
        conn = self.db.get_connection()
        try:
            now = _utc_now_iso()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO life_events (event_id, category, headline, sentiment, emotional_impact_json, occurred_at, mentioned_in_chat)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (event_id, category, headline, sentiment, json.dumps(emotional_impact or {}), now),
            )
        finally:
            conn.close()

    def get_recent_unmentioned_events(self, limit: int = 3) -> list[LifeEventRecord]:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM life_events WHERE mentioned_in_chat = 0 ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            )
            return [LifeEventRecord(**dict(r)) for r in cursor.fetchall()]
        finally:
            conn.close()

    def mark_event_mentioned(self, event_id: str) -> None:
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE life_events SET mentioned_in_chat = 1 WHERE event_id = ?", (event_id,))
        finally:
            conn.close()


def reset_conversation_memory(
    db: DatabaseManager | None = None,
    conversation_id: str | None = None,
    participant_handle: str | None = None,
) -> dict[str, int]:
    """Reset live conversation history, memories, relationship dynamics, and style profiles.

    If conversation_id or participant_handle is provided, only deletes data for matching threads.
    If both are None, resets ALL live conversation and memory tables in the database,
    restoring availability to default and clearing audit logs while strictly preserving
    all historical dataset tables, routine templates, and academic configurations.
    """
    db = db or get_db_manager()
    stats: dict[str, int] = {}

    with db.transaction() as cur:
        target_ids: list[str] = []
        if conversation_id:
            target_ids.append(conversation_id)
        if participant_handle:
            clean_handle = participant_handle.lstrip("@").strip().lower()
            cur.execute(
                """
                SELECT conversation_id FROM conversations 
                WHERE LOWER(participant_handle) = ? OR LOWER(participant_handle) = ? 
                   OR conversation_id = ? OR conversation_id = ?
                """,
                (f"@{clean_handle}", clean_handle, f"thread_{clean_handle}", clean_handle),
            )
            for row in cur.fetchall():
                c_id = row[0] if isinstance(row, (tuple, list)) else row["conversation_id"]
                if c_id not in target_ids:
                    target_ids.append(c_id)
            if f"thread_{clean_handle}" not in target_ids:
                target_ids.append(f"thread_{clean_handle}")

        if target_ids:
            placeholders = ",".join("?" for _ in target_ids)
            for table in ["memories", "messages", "relationship_states", "style_profiles"]:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE conversation_id IN ({placeholders})", target_ids)
                stats[table] = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT COUNT(*) FROM style_observations 
                WHERE profile_id IN (SELECT profile_id FROM style_profiles WHERE conversation_id IN ({placeholders}))
                """,
                target_ids,
            )
            stats["style_observations"] = cur.fetchone()[0]

            cur.execute(
                f"""
                DELETE FROM style_observations 
                WHERE profile_id IN (SELECT profile_id FROM style_profiles WHERE conversation_id IN ({placeholders}))
                """,
                target_ids,
            )
            cur.execute(f"DELETE FROM memories WHERE conversation_id IN ({placeholders})", target_ids)
            cur.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", target_ids)
            cur.execute(f"DELETE FROM relationship_states WHERE conversation_id IN ({placeholders})", target_ids)
            cur.execute(f"DELETE FROM style_profiles WHERE conversation_id IN ({placeholders})", target_ids)
            cur.execute(f"DELETE FROM conversations WHERE conversation_id IN ({placeholders})", target_ids)
            stats["conversations"] = len(target_ids)
        else:
            for table in [
                "messages",
                "memories",
                "relationship_states",
                "style_observations",
                "style_profiles",
                "conversations",
                "audit_events",
                "life_events",
            ]:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cur.fetchone()[0]
                cur.execute(f"DELETE FROM {table}")

            cur.execute(
                """
                UPDATE availability_state 
                SET current_state = 'AVAILABLE', 
                    current_activity = 'free_time', 
                    current_location = 'home', 
                    away_until = NULL, 
                    warning_sent_for_activity = NULL
                """
            )
            stats["availability_state_reset"] = 1

    return stats
