"""Unit tests for user style learning subsystem."""

import tempfile
from pathlib import Path
import pytest

from app.memory.style import StyleAnalyzer, StyleLearner
from app.storage.database import DatabaseManager
from app.storage.repositories import ConversationRepository


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_style.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


def test_style_analyzer_feature_extraction():
    text = "yo bro ngl this is wild fr..."
    features = StyleAnalyzer.analyze_message(text)

    assert features.word_count == 7
    assert features.is_all_lowercase is True
    assert features.is_all_caps is False
    assert features.has_trailing_ellipses is True
    assert "bro" in features.slang_words
    assert "ngl" in features.slang_words
    assert "fr" in features.slang_words


def test_style_analyzer_hinglish_and_emojis():
    text = "arre yaar what is the scene? 😂🔥"
    features = StyleAnalyzer.analyze_message(text)

    assert "arre" in features.hinglish_words
    assert "yaar" in features.hinglish_words
    assert "scene" in features.hinglish_words
    assert features.emoji_count >= 1


def test_style_learner_accumulation_and_prompt_notes(temp_db):
    conv_repo = ConversationRepository(temp_db)
    conv_repo.get_or_create("thread_style_acc", participant_handle="@test_slang_user")

    learner = StyleLearner(temp_db)

    # 1. Single observation -> weak signal (< 0.5 confidence)
    learner.observe_message("thread_style_acc", "bro nice")
    notes_1 = learner.get_style_notes_for_prompt("thread_style_acc")
    # Terse message recognized
    assert any("terse" in n.lower() for n in notes_1)
    # Slang is not yet high confidence (evidence_count = 1 -> confidence = 0.25)
    assert not any("frequently uses informal slang" in n for n in notes_1)

    # 2. Repeated observations -> confidence reaches threshold (>= 0.5)
    learner.observe_message("thread_style_acc", "bro why")
    learner.observe_message("thread_style_acc", "bro stop")
    notes_2 = learner.get_style_notes_for_prompt("thread_style_acc")

    assert any("frequently uses informal slang" in n for n in notes_2)
    assert any("bro" in n for n in notes_2)
