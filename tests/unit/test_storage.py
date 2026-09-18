"""Unit and integration tests for SQLite persistence and repositories."""

import os
import tempfile
from pathlib import Path
import pytest

from app.storage.database import DatabaseManager
from app.storage.repositories import (
    AuditRepository,
    ConversationRepository,
    MemoryRepository,
    MessageRepository,
    RelationshipRepository,
    StyleRepository,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_agent.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


def test_schema_initialization_and_pragmas(temp_db):
    conn = temp_db.get_connection()
    try:
        cursor = conn.cursor()
        # Verify WAL mode
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].upper()
        assert journal_mode == "WAL"

        # Verify foreign keys
        cursor.execute("PRAGMA foreign_keys;")
        foreign_keys = cursor.fetchone()[0]
        assert foreign_keys == 1
    finally:
        conn.close()


def test_conversation_repository(temp_db):
    repo = ConversationRepository(temp_db)
    conv = repo.get_or_create("thread_123", participant_handle="@testuser", participant_name="Test User")
    assert conv.conversation_id == "thread_123"
    assert conv.participant_handle == "@testuser"
    assert conv.status == "ACTIVE"

    # Fetching again returns existing record
    conv2 = repo.get("thread_123")
    assert conv2 is not None
    assert conv2.participant_name == "Test User"


def test_message_repository_lifecycle_and_idempotency(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_abc", participant_handle="@user1")

    msg_repo = MessageRepository(temp_db)
    fp = "hash_unique_message_1"

    assert not msg_repo.exists_by_fingerprint(fp)

    msg = msg_repo.record_incoming_message("thread_abc", fp, "@user1", "Hello there!")
    assert msg.status == "RECEIVED"
    assert msg.sender_type == "USER"
    assert msg_repo.exists_by_fingerprint(fp)

    # Update status to PROCESSING and SENT
    msg_repo.update_status(msg.message_id, "PROCESSING")
    recent = msg_repo.get_recent_messages("thread_abc", limit=5)
    assert len(recent) == 1
    assert recent[0].status == "PROCESSING"

    # Record character response
    char_msg = msg_repo.record_character_message("thread_abc", "General Kenobi.")
    assert char_msg.sender_type == "CHARACTER"

    # Sliding window should now contain both messages in chronological order
    window = msg_repo.get_recent_messages("thread_abc", limit=10)
    assert len(window) == 2
    assert window[0].content == "Hello there!"
    assert window[1].content == "General Kenobi."


def test_memory_repository(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_mem", participant_handle="@user_mem")

    mem_repo = MemoryRepository(temp_db)
    m = mem_repo.add_memory("thread_mem", "FACT", "User works as a software architect", confidence=0.9)
    assert m.statement == "User works as a software architect"
    assert m.access_count == 0

    mem_repo.touch_memory(m.memory_id)
    memories = mem_repo.get_memories_for_conversation("thread_mem")
    assert len(memories) == 1
    assert memories[0].access_count == 1


def test_relationship_repository(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_rel", participant_handle="@user_rel")

    rel_repo = RelationshipRepository(temp_db)
    rel = rel_repo.get_or_create("thread_rel")
    assert rel.familiarity == 0.1
    assert rel.playfulness == 0.5

    rel_repo.update_chemistry("thread_rel", familiarity=0.4, playfulness=0.8, sarcasm_tolerance=0.7, trust=0.5)
    updated = rel_repo.get_or_create("thread_rel")
    assert updated.familiarity == 0.4
    assert updated.playfulness == 0.8


def test_style_repository(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_style", participant_handle="@user_style")

    style_repo = StyleRepository(temp_db)
    profile = style_repo.get_or_create_profile("thread_style")
    assert profile.avg_sentence_length == 8.0

    style_repo.record_observation(profile.profile_id, "slang", "bro", confidence=0.3)
    obs = style_repo.get_observations(profile.profile_id)
    assert len(obs) == 1
    assert obs[0].evidence_count == 1

    # Second observation updates evidence count
    style_repo.record_observation(profile.profile_id, "slang", "bro", confidence=0.6)
    obs2 = style_repo.get_observations(profile.profile_id)
    assert len(obs2) == 1
    assert obs2[0].evidence_count == 2
    assert obs2[0].confidence == 0.6


def test_audit_repository(temp_db):
    audit_repo = AuditRepository(temp_db)
    event_id = audit_repo.record_event(
        event_type="TEST_EVENT",
        severity="INFO",
        component="TEST",
        payload={"key": "value"},
        correlation_id="corr-999",
    )
    assert event_id is not None

    events = audit_repo.get_recent_events(limit=10)
    assert len(events) >= 1
    assert events[0]["event_type"] == "TEST_EVENT"
    assert events[0]["correlation_id"] == "corr-999"
