"""Unit tests for Humor Engine."""

import json
import tempfile
from pathlib import Path
import pytest

from app.humor.engine import HumorEngine
from app.humor.retrieval import HumorRetriever
from app.storage.database import DatabaseManager
from app.storage.repositories import ConversationRepository, RelationshipRepository


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_humor.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


def test_humor_exemplar_seeding_and_retrieval(temp_db):
    retriever = HumorRetriever(temp_db)
    retriever.seed_exemplars_if_empty()

    exemplar = retriever.get_exemplar("dry_understated", max_intensity=0.5)
    assert exemplar is not None
    assert exemplar.category == "dry_understated"
    assert len(exemplar.exemplar_response) > 0


def test_humor_suppression_on_distress(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_distress", participant_handle="@sad_user")

    engine = HumorEngine(temp_db)
    decision = engine.evaluate_humor("thread_distress", "I am so heartbroken, my grandmother passed away today.")

    assert decision.is_humor_appropriate is False
    assert decision.intensity == 0.0
    assert "SUPPRESS ALL HUMOR" in decision.directive_text


def test_humor_intensity_modulation(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_stranger", participant_handle="@new_user")

    rel_repo = RelationshipRepository(temp_db)
    # Low chemistry (stranger)
    rel_repo.update_chemistry("thread_stranger", familiarity=0.1, playfulness=0.2, sarcasm_tolerance=0.1, trust=0.1)

    engine = HumorEngine(temp_db)
    decision_stranger = engine.evaluate_humor("thread_stranger", "Hello there!")
    # I_max = 0.2 + (0.5 * 0.1) + (0.3 * 0.1) = 0.28
    assert decision_stranger.intensity <= 0.35

    # High chemistry (close friend)
    conv_repo.get_or_create("thread_friend", participant_handle="@best_friend")
    rel_repo.update_chemistry("thread_friend", familiarity=0.9, playfulness=0.9, sarcasm_tolerance=0.9, trust=0.9)

    decision_friend = engine.evaluate_humor("thread_friend", "Look at this terrible code I wrote")
    # I_max = 0.2 + (0.5 * 0.9) + (0.3 * 0.9) = 0.92
    assert decision_friend.intensity >= 0.8
    assert decision_friend.category == "sarcastic_teasing"


def test_humor_callback_trigger(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_callback", participant_handle="@callback_user")

    rel_repo = RelationshipRepository(temp_db)
    jokes = [
        {
            "callback_id": "joke_broken_chair",
            "trigger_keywords": ["chair", "ergonomic", "desk"],
            "punchline_hint": "ask if their chair collapsed again",
        }
    ]
    rel_repo.update_chemistry(
        "thread_callback",
        familiarity=0.5,
        playfulness=0.5,
        sarcasm_tolerance=0.5,
        trust=0.5,
        running_jokes_json=json.dumps(jokes),
    )

    engine = HumorEngine(temp_db)
    decision = engine.evaluate_humor("thread_callback", "I just bought a new desk setup.")

    assert decision.callback_to_reference == "joke_broken_chair"
    assert "Callback Reference:" in decision.directive_text
    assert "chair collapsed" in decision.directive_text
