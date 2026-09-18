"""Unit tests for Multi-Tier Memory Engine."""

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from app.memory.manager import MemoryManager
from app.memory.retrieval import compute_memory_score, rank_and_filter_memories
from app.storage.database import DatabaseManager
from app.storage.repositories import ConversationRepository, MemoryRecord


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_memory.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


def test_compute_memory_score_factors():
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=20)).isoformat()

    recent_mem = MemoryRecord(
        memory_id="m1",
        conversation_id="c1",
        memory_type="FACT",
        statement="User is a backend engineer in Seattle",
        confidence=1.0,
        access_count=5,
        created_at=now.isoformat(),
        last_accessed_at=now.isoformat(),
    )

    old_mem = MemoryRecord(
        memory_id="m2",
        conversation_id="c1",
        memory_type="FACT",
        statement="User is a backend engineer in Seattle",
        confidence=1.0,
        access_count=0,
        created_at=old_time,
        last_accessed_at=old_time,
    )

    query = "What kind of engineering do you do?"

    score_recent = compute_memory_score(recent_mem, query, now_utc=now)
    score_old = compute_memory_score(old_mem, query, now_utc=now)

    assert score_recent > score_old
    assert score_recent >= 0.5


def test_rank_and_filter_memories():
    now = datetime.now(timezone.utc).isoformat()
    mems = [
        MemoryRecord("1", "c1", "FACT", "User loves espresso", 1.0, 5, now, now),
        MemoryRecord("2", "c1", "FACT", "User dislikes loud music", 0.8, 1, now, now),
        MemoryRecord("3", "c1", "PREFERENCE", "User works in finance", 0.9, 3, now, now),
    ]

    # Query matching espresso
    matched = rank_and_filter_memories(mems, "Can you make me an espresso?", threshold=0.3)
    assert len(matched) >= 1
    assert matched[0].memory_id == "1"


def test_memory_manager_extraction_and_retrieval(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_test_mem", participant_handle="@dev_dan")

    mgr = MemoryManager(temp_db)

    # 1. Extract facts from user message
    extracted = mgr.extract_and_persist_facts(
        "thread_test_mem",
        "Honestly I work as a distributed systems architect and I live in Tokyo.",
    )
    assert len(extracted) == 2
    statements = [e.statement for e in extracted]
    assert any("distributed systems architect" in s for s in statements)
    assert any("Tokyo" in s for s in statements)

    # 2. Extract again with same message -> deduplication prevents duplicate insert
    extracted2 = mgr.extract_and_persist_facts(
        "thread_test_mem",
        "I live in Tokyo.",
    )
    assert len(extracted2) == 0

    # 3. Retrieve relevant memories
    relevant = mgr.get_relevant_memories("thread_test_mem", "How is the weather in Tokyo?")
    assert len(relevant) >= 1
    assert any("Tokyo" in r for r in relevant)
