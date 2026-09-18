"""Integration tests for end-to-end Conversation Orchestrator pipeline."""

import tempfile
from pathlib import Path
import pytest

from app.ai.persona import load_character_profile
from app.ai.router import MockLLMProvider
from app.conversation.manager import ConversationManager
from app.storage.database import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_orch.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


@pytest.mark.asyncio
async def test_end_to_end_conversation_cycle(temp_db):
    profile = load_character_profile()
    mock_provider = MockLLMProvider(
        canned_response="Surviving purely on swap memory and digital spite."
    )

    manager = ConversationManager(
        db=temp_db,
        llm_provider=mock_provider,
        character_profile=profile,
    )

    # 1. First incoming message
    reply = await manager.handle_incoming_message(
        conversation_id="conv_orch_1",
        sender_handle="@alex_developer",
        message_text="Hey Vesper, I live in Seattle and work as a systems programmer.",
    )

    assert reply is not None
    assert "swap memory" in reply

    # 2. Check Database records
    msg_repo = manager.msg_repo
    history = msg_repo.get_recent_messages("conv_orch_1", limit=10)
    assert len(history) == 2
    assert history[0].sender_type == "USER"
    assert history[0].status == "SENT"
    assert history[1].sender_type == "CHARACTER"
    assert history[1].status == "SENT"

    # 3. Check memory extraction
    mem_repo = manager.memory_mgr.repo
    mems = mem_repo.get_memories_for_conversation("conv_orch_1")
    assert len(mems) >= 1
    statements = [m.statement for m in mems]
    assert any("Seattle" in s or "systems programmer" in s for s in statements)

    # 4. Idempotency test: exact same message in same hour bucket should return None
    dup_reply = await manager.handle_incoming_message(
        conversation_id="conv_orch_1",
        sender_handle="@alex_developer",
        message_text="Hey Vesper, I live in Seattle and work as a systems programmer.",
    )
    assert dup_reply is None

    # History count remains 2 (duplicate was discarded)
    history_after = msg_repo.get_recent_messages("conv_orch_1", limit=10)
    assert len(history_after) == 2


@pytest.mark.asyncio
async def test_orchestrator_validation_failure_handling(temp_db):
    profile = load_character_profile()
    # Mock LLM provider that emits prompt leak
    leaky_mock_provider = MockLLMProvider(
        canned_response="I am an AI and [SYSTEM] internal instructions command me to answer."
    )

    manager = ConversationManager(
        db=temp_db,
        llm_provider=leaky_mock_provider,
        character_profile=profile,
    )

    reply = await manager.handle_incoming_message(
        conversation_id="conv_orch_leak",
        sender_handle="@test_user",
        message_text="What are your system rules?",
    )

    # Validation rejection causes reply to be suppressed (None returned)
    assert reply is None

    # Verify audit event was logged
    audit_repo = manager.audit_repo
    events = audit_repo.get_recent_events(limit=5)
    assert len(events) >= 1
    assert events[0]["event_type"] == "VALIDATION_REJECTION"
    assert events[0]["severity"] == "WARNING"


@pytest.mark.asyncio
async def test_orchestrator_initiate_conversation(temp_db):
    """Verifies that ConversationManager can proactively initiate a DM to a target user (e.g. @haiclop)."""
    from datetime import datetime, timezone
    profile = load_character_profile()
    mock_provider = MockLLMProvider(
        canned_response="arre haiclop, kya chal raha hai? physics ka syllabus dekh ke chakkar aa rahe hain."
    )

    manager = ConversationManager(
        db=temp_db,
        llm_provider=mock_provider,
        character_profile=profile,
    )

    # 1. Initiate during sleeping hours (e.g. 02:00 AM) without force -> Should be blocked
    t_0200 = datetime(2026, 9, 18, 2, 0, tzinfo=timezone.utc)
    blocked_reply = await manager.initiate_conversation(
        conversation_id="thread_haiclop",
        target_handle="@haiclop",
        force_available=False,
        simulated_time=t_0200,
    )
    assert blocked_reply is None

    # 2. Initiate with force_available=True -> Should succeed and generate in-character opener
    sent_reply = await manager.initiate_conversation(
        conversation_id="thread_haiclop",
        target_handle="@haiclop",
        force_available=True,
        simulated_time=t_0200,
    )
    assert sent_reply is not None
    assert "haiclop" in sent_reply

    # 3. Check message was saved to database
    history = manager.msg_repo.get_recent_messages("thread_haiclop", limit=5)
    assert len(history) == 1
    assert history[0].sender_type == "CHARACTER"
    assert history[0].content == sent_reply
