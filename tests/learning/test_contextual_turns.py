"""Tests for contextual turn extraction, behavioral patterns, and situation retrieval."""

import tempfile
from pathlib import Path
import pytest

from app.learning.models import MessageType, NormalizedConversation, NormalizedMessage
from app.learning.normalizer import InstagramNormalizer
from app.conversation.situation_retriever import SituationAwareRetriever
from app.conversation.social_intent import SocialIntentAnalyzer
from app.storage.database import DatabaseManager
from app.storage.historical_repo import HistoricalRepository


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = DatabaseManager(db_path)
    db.initialize_schema()
    yield db
    try:
        db_path.unlink()
    except Exception:
        pass


def test_chronological_normalization_and_turn_extraction():
    msgs = [
        NormalizedMessage(
            conversation_id="conv_test",
            sender_id="UserA",
            sender_display_name="UserA",
            timestamp_ms=1000,
            timestamp_iso="2026-09-18T10:00:00Z",
            message_type=MessageType.TEXT,
            text="yo kya scene",
            sender_role="contact",
            message_index=0,
        ),
        NormalizedMessage(
            conversation_id="conv_test",
            sender_id="Arnav",
            sender_display_name="Arnav Srivastava",
            timestamp_ms=2000,
            timestamp_iso="2026-09-18T10:00:02Z",
            message_type=MessageType.TEXT,
            text="kuch nahi valorant",
            sender_role="operator",
            message_index=1,
        ),
        NormalizedMessage(
            conversation_id="conv_test",
            sender_id="UserA",
            sender_display_name="UserA",
            timestamp_ms=3000,
            timestamp_iso="2026-09-18T10:00:03Z",
            message_type=MessageType.TEXT,
            text="Why are you so dumb",
            sender_role="contact",
            message_index=2,
        ),
        NormalizedMessage(
            conversation_id="conv_test",
            sender_id="Arnav",
            sender_display_name="Arnav Srivastava",
            timestamp_ms=4000,
            timestamp_iso="2026-09-18T10:00:04Z",
            message_type=MessageType.TEXT,
            text="tu chup reh",
            sender_role="operator",
            message_index=3,
        ),
    ]

    conv = NormalizedConversation(
        conversation_id="conv_test",
        file_path="fake.json",
        participants=["Arnav Srivastava", "UserA"],
        operator_name="Arnav Srivastava",
        contact_name="UserA",
        messages=msgs,
        message_count=len(msgs),
    )

    turns = InstagramNormalizer.extract_contextual_turns(
        conv,
        operator_name="Arnav Srivastava",
        contact_id="user_a",
    )

    assert len(turns) == 2
    # Turn 0: greeting
    assert turns[0].contact_text == "yo kya scene"
    assert turns[0].operator_text == "kuch nahi valorant"

    # Turn 1: playful banter
    assert turns[1].contact_text == "Why are you so dumb"
    assert turns[1].operator_text == "tu chup reh"
    assert turns[1].social_act == "playful_insult"
    assert turns[1].response_strategy == "playful_counter"
    assert turns[1].response_length_category == "very_short"


def test_repository_turns_and_situation_retriever(temp_db):
    repo = HistoricalRepository(temp_db)
    msgs = [
        NormalizedMessage(
            conversation_id="c1",
            sender_id="U1",
            sender_display_name="U1",
            timestamp_ms=1000,
            timestamp_iso="",
            message_type=MessageType.TEXT,
            text="Why are you so dumb",
            sender_role="contact",
        ),
        NormalizedMessage(
            conversation_id="c1",
            sender_id="Op",
            sender_display_name="Arnav Srivastava",
            timestamp_ms=2000,
            timestamp_iso="",
            message_type=MessageType.TEXT,
            text="tu chup reh",
            sender_role="operator",
        ),
    ]
    conv = NormalizedConversation(
        conversation_id="c1",
        file_path="",
        participants=["Arnav Srivastava", "U1"],
        operator_name="Arnav Srivastava",
        contact_name="U1",
        messages=msgs,
        message_count=2,
    )

    turns = InstagramNormalizer.extract_contextual_turns(conv, operator_name="Arnav Srivastava", contact_id="u1")
    count = repo.batch_insert_turns(turns)
    assert count == 1

    stored_turns = repo.get_turns(social_act="playful_insult")
    assert len(stored_turns) == 1
    assert stored_turns[0]["contact_text"] == "Why are you so dumb"

    # Test SituationAwareRetriever
    retriever = SituationAwareRetriever(temp_db, repo=repo)
    intent = SocialIntentAnalyzer.analyze("Why are you so dumb")
    results = retriever.retrieve_examples("Why are you so dumb", intent, contact_id="u1", limit=2)
    assert len(results) == 1
    assert results[0]["operator_text"] == "tu chup reh"
    assert results[0]["score"] > 0.6
