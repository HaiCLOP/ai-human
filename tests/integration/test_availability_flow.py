"""End-to-end acceptance integration tests for routine, departure, silent queueing, and batch return."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from app.ai.persona import load_character_profile
from app.ai.router import MockLLMProvider
from app.conversation.manager import ConversationManager
from app.routine.manager import RoutineManager
from app.routine.models import AvailabilityState
from app.storage.database import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_avail_flow.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


@pytest.mark.asyncio
async def test_section_24_acceptance_scenario(temp_db):
    """Verifies the complete Section 24 acceptance scenario:
    1. 16:45: Normal active conversation.
    2. 17:00: Tuition begins -> Natural departure goodbye generated, away_until set to 19:00.
    3. 17:30: User sends message while she is away -> Message persisted in DB in 'RECEIVED' state, zero LLM calls, returns None.
    4. 19:00: Tuition ends -> Character becomes AVAILABLE, batches queued messages, and returns with a single natural reply.
    """
    profile = load_character_profile()
    mock_llm = MockLLMProvider(
        canned_response="haan main theek hoon, abhi ghar pe chill kar rahi thi."
    )

    routine_mgr = RoutineManager(db=temp_db, routine_config=profile.routine)
    conv_mgr = ConversationManager(
        db=temp_db,
        llm_provider=mock_llm,
        character_profile=profile,
        routine_manager=routine_mgr,
    )

    conv_id = "test_acceptance_thread"
    user_handle = "@test_friend"

    # -------------------------------------------------------------------------
    # STEP 1: 16:45 - Normal active conversation
    # -------------------------------------------------------------------------
    t_1645 = datetime(2026, 9, 21, 16, 45, tzinfo=timezone.utc)  # Monday 16:45 (self_study / available)
    reply_1645 = await conv_mgr.handle_incoming_message(
        conversation_id=conv_id,
        sender_handle=user_handle,
        message_text="kya chal raha hai?",
        simulated_time=t_1645,
    )
    assert reply_1645 is not None
    assert "main theek hoon" in reply_1645

    # -------------------------------------------------------------------------
    # STEP 2: 17:00 - Tuition begins -> Triggers natural departure
    # -------------------------------------------------------------------------
    t_1700 = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)  # Monday 17:00 (maths_tuition begins)
    mock_llm.canned_response = "chalo mujhe tuition ke liye nikalna padega 😭 2 ghante baad aati hoon, maths ka paper hai parso 💀"

    reply_1700 = await conv_mgr.handle_incoming_message(
        conversation_id=conv_id,
        sender_handle=user_handle,
        message_text="ek aur cheez batao",
        simulated_time=t_1700,
    )
    assert reply_1700 is not None
    assert "tuition" in reply_1700

    # Assert internal state updated in database
    avail_rec = conv_mgr.routine_mgr.avail_repo.get_or_create("default")
    assert avail_rec.current_state == "AT_TUITION"
    assert avail_rec.current_activity == "maths_tuition"
    assert avail_rec.away_until is not None
    assert "19:00" in avail_rec.away_until

    # -------------------------------------------------------------------------
    # STEP 3: 17:30 - Message received while character is away at tuition
    # -------------------------------------------------------------------------
    t_1730 = datetime(2026, 9, 21, 17, 30, tzinfo=timezone.utc)  # Monday 17:30 (during tuition)
    reply_1730 = await conv_mgr.handle_incoming_message(
        conversation_id=conv_id,
        sender_handle=user_handle,
        message_text="where are you? free kab hogi?",
        simulated_time=t_1730,
    )
    # MUST BE SILENT: Zero immediate reply returned
    assert reply_1730 is None

    # Verify message was persisted in database in 'RECEIVED' state
    history = conv_mgr.msg_repo.get_recent_messages(conv_id, limit=10)
    msg_1730 = [m for m in history if "where are you" in m.content]
    assert len(msg_1730) == 1
    assert msg_1730[0].status == "RECEIVED"

    # User sends another message at 18:30
    t_1830 = datetime(2026, 9, 21, 18, 30, tzinfo=timezone.utc)
    reply_1830 = await conv_mgr.handle_incoming_message(
        conversation_id=conv_id,
        sender_handle=user_handle,
        message_text="hello? reply toh de",
        simulated_time=t_1830,
    )
    assert reply_1830 is None

    # -------------------------------------------------------------------------
    # STEP 4: 19:00 - Tuition ends -> Character becomes AVAILABLE & returns
    # -------------------------------------------------------------------------
    t_1905 = datetime(2026, 9, 21, 19, 5, tzinfo=timezone.utc)  # Monday 19:05 (rest_and_snacks / AVAILABLE)
    state_1905, act_1905 = conv_mgr.routine_mgr.resolve_availability(t_1905)
    assert state_1905 == AvailabilityState.AVAILABLE

    mock_llm.canned_response = "backkk 😭 tuition ne jaan le li meri. haan abhi dekha sab, kya chal raha tha?"

    reply_1905 = await conv_mgr.handle_incoming_message(
        conversation_id=conv_id,
        sender_handle=user_handle,
        message_text="are you back yet?",
        simulated_time=t_1905,
    )

    assert reply_1905 is not None
    assert "backkk" in reply_1905

    # Verify that the queued messages from 17:30 and 18:30 are now marked as SENT / processed
    history_final = conv_mgr.msg_repo.get_recent_messages(conv_id, limit=15)
    unhandled = [m for m in history_final if m.sender_type == "USER" and m.status == "RECEIVED"]
    assert len(unhandled) == 0


@pytest.mark.asyncio
async def test_tuition_extended_override(temp_db):
    """Verifies that if tuition runs late via away_until update, character remains unavailable until new time."""
    profile = load_character_profile()
    mock_llm = MockLLMProvider()

    routine_mgr = RoutineManager(db=temp_db, routine_config=profile.routine)
    conv_mgr = ConversationManager(
        db=temp_db,
        llm_provider=mock_llm,
        character_profile=profile,
        routine_manager=routine_mgr,
    )

    # Tuition ran late: operator or life event extends away_until to 19:20 (within 19:00-19:30 rest slot)
    t_1920_iso = datetime(2026, 9, 21, 19, 20, tzinfo=timezone.utc).isoformat()
    routine_mgr.mark_departure("default", "maths_tuition_extra", AvailabilityState.AT_TUITION, "tuition", t_1920_iso)

    # Check at 19:15 -> Should still be AT_TUITION (even though 19:00 scheduled tuition ended)
    t_1915 = datetime(2026, 9, 21, 19, 15, tzinfo=timezone.utc)
    state, act = routine_mgr.resolve_availability(t_1915)
    assert state == AvailabilityState.AT_TUITION

    # Check at 19:25 -> Away window expired (19:20 < 19:25 < 19:30 rest_and_snacks) -> AVAILABLE
    t_1925 = datetime(2026, 9, 21, 19, 25, tzinfo=timezone.utc)
    state_after, act_after = routine_mgr.resolve_availability(t_1925)
    assert state_after == AvailabilityState.AVAILABLE
