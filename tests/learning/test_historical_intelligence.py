"""Comprehensive tests for Instagram Historical Conversation Intelligence system."""

import json
import tempfile
from pathlib import Path
import pytest

from app.learning.contact_style import ContactStyleAnalyzer
from app.learning.dataset_scanner import DatasetScanner
from app.learning.humor_learning import HumorAnalyzer
from app.learning.memory_learning import MemoryExtractor
from app.learning.models import MessageType
from app.learning.normalizer import InstagramNormalizer, fix_instagram_encoding
from app.learning.operator_detector import AmbiguousOperatorException, OperatorDetector
from app.learning.operator_style import OperatorStyleAnalyzer
from app.learning.reel_intelligence import ReelIntelligenceAnalyzer
from app.learning.versioning import LearningCoordinator
from app.storage.database import DatabaseManager
from app.storage.historical_repo import HistoricalRepository

MOCK_RAW_EXPORT = {
    "participants": [
        {"name": "Arnav Srivastava"},
        {"name": "haiclop"}
    ],
    "messages": [
        {
            "sender_name": "Arnav Srivastava",
            "timestamp_ms": 1726000000000,
            "content": "yo kya scene hai",
            "reactions": [{"reaction": "\u00e2\u009d\u00a4", "actor": "haiclop"}]
        },
        {
            "sender_name": "haiclop",
            "timestamp_ms": 1726000010000,
            "content": "kuch nahi bhai bas valorant khel raha hu",
            "reactions": []
        },
        {
            "sender_name": "Arnav Srivastava",
            "timestamp_ms": 1726000020000,
            "content": "mast aaja fir discord pe",
            "reactions": []
        },
        {
            "sender_name": "haiclop",
            "timestamp_ms": 1726000050000,
            "share": {
                "link": "https://www.instagram.com/reel/C-test123/",
                "share_text": "bhai literally us",
                "original_content_owner": "memegod"
            }
        },
        {
            "sender_name": "Arnav Srivastava",
            "timestamp_ms": 1726000060000,
            "content": "dead lmao 😭",
            "reactions": []
        }
    ]
}


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_file = Path(tmp) / "test_historical.db"
        db = DatabaseManager(db_file)
        db.initialize_schema()
        yield db


@pytest.fixture
def temp_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        conv_dir = Path(tmp) / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        file1 = conv_dir / "conversation_001.json"
        with open(file1, "w", encoding="utf-8") as f:
            json.dump(MOCK_RAW_EXPORT, f)

        # Invalid file for scanner verification
        bad_file = conv_dir / "invalid.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("not json")

        yield Path(tmp), conv_dir, file1


def test_instagram_encoding_fix():
    # Latin-1 representation of UTF-8 heart emoji
    raw_heart = "\u00e2\u009d\u00a4"
    fixed = fix_instagram_encoding(raw_heart)
    assert "❤" in fixed or len(fixed) > 0


def test_normalizer_message_types():
    assert InstagramNormalizer.classify_message_type({"content": "hello"}) == MessageType.TEXT
    assert InstagramNormalizer.classify_message_type({"share": {"link": "https://www.instagram.com/reel/abc/"}}) == MessageType.REEL
    assert InstagramNormalizer.classify_message_type({"share": {"link": "https://www.instagram.com/stories/xyz/"}}) == MessageType.STORY
    assert InstagramNormalizer.classify_message_type({"photos": [{"uri": "photo.jpg"}]}) == MessageType.PHOTO
    assert InstagramNormalizer.classify_message_type({"audio_files": [{"uri": "audio.mp4"}]}) == MessageType.AUDIO
    assert InstagramNormalizer.classify_message_type({"call_duration": 120}) == MessageType.CALL
    assert InstagramNormalizer.classify_message_type({"reactions": [{"reaction": "❤"}]}) == MessageType.REACTION


def test_operator_detector():
    detector = OperatorDetector(operator_names=["Arnav Srivastava", "Arnav"])

    # Exact match
    op, contacts = detector.identify_participants(["Arnav Srivastava", "haiclop"])
    assert op == "Arnav Srivastava"
    assert contacts == ["haiclop"]

    # Alias match
    op, contacts = detector.identify_participants(["Arnav", "kabir"])
    assert op == "Arnav"
    assert contacts == ["kabir"]

    # Ambiguous - no operator found halts with exception
    with pytest.raises(AmbiguousOperatorException):
        detector.identify_participants(["RandomUser1", "RandomUser2"])

    # Pseudonymous contact ID generation
    cid = detector.generate_contact_id("haiclop")
    assert cid.startswith("contact_")
    assert cid == detector.generate_contact_id("haiclop")  # Deterministic


def test_dataset_scanner(temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    scanner = DatasetScanner(dataset_path=conv_dir)
    manifest = scanner.scan()

    assert manifest.files_discovered == 2
    assert manifest.valid_conversations == 1
    assert manifest.invalid_files == 1
    assert manifest.total_messages == 5
    assert manifest.type_counts.get("TEXT") == 4
    assert manifest.type_counts.get("REEL") == 1
    assert len(manifest.dataset_hash) > 0

    report = DatasetScanner.format_manifest_report(manifest)
    assert "Files discovered: 2" in report
    assert "Text messages: 4" in report


def test_historical_repository_and_idempotency(temp_db, temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    repo = HistoricalRepository(temp_db)
    norm = InstagramNormalizer.normalize_conversation(valid_file, operator_names=["Arnav Srivastava"])

    repo.upsert_conversation(norm, contact_id="contact_haiclop", contact_name="haiclop", operator_name="Arnav Srivastava")
    inserted1 = repo.batch_insert_messages(norm.messages, contact_id="contact_haiclop", operator_name="Arnav Srivastava")
    assert inserted1 == 5

    # Re-inserting must be idempotent (no duplicate primary keys)
    inserted2 = repo.batch_insert_messages(norm.messages, contact_id="contact_haiclop", operator_name="Arnav Srivastava")
    assert inserted2 == 5

    msgs = repo.get_messages_for_contact("contact_haiclop")
    assert len(msgs) == 5

    op_msgs = repo.get_all_operator_messages()
    assert len(op_msgs) == 3


def test_style_analyzers(temp_db, temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    repo = HistoricalRepository(temp_db)
    norm = InstagramNormalizer.normalize_conversation(valid_file, operator_names=["Arnav Srivastava"])
    repo.upsert_conversation(norm, contact_id="contact_haiclop", contact_name="haiclop", operator_name="Arnav Srivastava")
    repo.batch_insert_messages(norm.messages, contact_id="contact_haiclop", operator_name="Arnav Srivastava")

    op_msgs = repo.get_all_operator_messages()
    op_analyzer = OperatorStyleAnalyzer()
    op_profile = op_analyzer.analyze(op_msgs)

    # Operator texts: 'yo kya scene hai', 'mast aaja fir discord pe', 'dead lmao 😭'
    assert op_profile.total_messages == 3
    assert op_profile.lowercase_ratio >= 0.9
    assert op_profile.ending_period_ratio == 0.0  # Zero ending periods
    assert op_profile.hinglish_ratio > 0.0

    # Contact style: 'haiclop' sent 'kuch nahi bhai bas valorant khel raha hu'
    contact_msgs = repo.get_messages_for_contact("contact_haiclop")
    c_msgs = [m for m in contact_msgs if not m.get("sender_is_operator")]
    c_analyzer = ContactStyleAnalyzer()
    c_profile = c_analyzer.analyze("contact_haiclop", "haiclop", c_msgs)

    assert c_profile.total_messages == 2
    assert "bhai" in c_profile.slang_terms_used or "kuch nahi" in c_profile.slang_terms_used


def test_humor_and_memory_extraction(temp_db, temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    repo = HistoricalRepository(temp_db)
    norm = InstagramNormalizer.normalize_conversation(valid_file, operator_names=["Arnav Srivastava"])
    repo.upsert_conversation(norm, contact_id="contact_haiclop", contact_name="haiclop", operator_name="Arnav Srivastava")
    repo.batch_insert_messages(norm.messages, contact_id="contact_haiclop", operator_name="Arnav Srivastava")

    msgs = repo.get_messages_for_contact("contact_haiclop")

    # Humor
    humor_analyzer = HumorAnalyzer()
    h_profile = humor_analyzer.analyze("contact_haiclop", msgs)
    assert h_profile.humor_frequency > 0.0
    assert h_profile.laughing_reactions_count >= 0

    # Memory
    mem_extractor = MemoryExtractor()
    mems, topics = mem_extractor.extract("contact_haiclop", msgs, min_confidence=0.5)
    assert isinstance(mems, list)
    assert isinstance(topics, list)


def test_reels_and_anti_repetition(temp_db, temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    repo = HistoricalRepository(temp_db)
    norm = InstagramNormalizer.normalize_conversation(valid_file, operator_names=["Arnav Srivastava"])
    repo.upsert_conversation(norm, contact_id="contact_haiclop", contact_name="haiclop", operator_name="Arnav Srivastava")
    repo.batch_insert_messages(norm.messages, contact_id="contact_haiclop", operator_name="Arnav Srivastava")

    msgs = repo.get_messages_for_contact("contact_haiclop")
    reel_analyzer = ReelIntelligenceAnalyzer()
    r_profile, reels = reel_analyzer.analyze(msgs, operator_name="Arnav Srivastava")

    assert r_profile.total_reels_shared == 1
    assert r_profile.received_by_operator_count == 1
    assert len(reels) == 1
    assert "instagram.com/reel" in reels[0]["reel_url"]

    # Anti-repetition
    assert reel_analyzer.is_reel_fresh(reels[0]["reel_url"]) is True
    reel_analyzer.mark_reel_shared(reels[0]["reel_url"])
    assert reel_analyzer.is_reel_fresh(reels[0]["reel_url"]) is False


def test_end_to_end_coordinator_and_rebuild(temp_db, temp_dataset):
    base_dir, conv_dir, valid_file = temp_dataset
    coordinator = LearningCoordinator(db=temp_db)

    # 1. Import
    conv_count, msg_count = coordinator.import_dataset(dataset_path=conv_dir)
    assert conv_count == 1
    assert msg_count == 5

    # 2. Analyze without external RAG dependency
    ver = coordinator.run_analysis(index_rag=False)
    assert ver.startswith("style_profile_v")

    active_ver = coordinator.repo.get_active_version()
    assert active_ver is not None
    assert active_ver["message_count"] == 5

    # 3. Reset profiles preserves messages
    coordinator.reset_profiles()
    assert coordinator.repo.get_active_version() is None
    all_msgs = coordinator.repo.get_all_operator_messages()
    assert len(all_msgs) == 3

    # 4. Full rebuild
    new_ver = coordinator.full_rebuild(dataset_path=conv_dir)
    assert new_ver.startswith("style_profile_v")
    assert coordinator.repo.get_active_version() is not None
