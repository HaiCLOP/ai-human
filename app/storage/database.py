"""SQLite database connection manager and schema migration engine."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from app.core.config import get_settings
from app.core.exceptions import DatabaseException
from app.core.logging import get_logger

logger = get_logger("storage.database")

DDL_SCHEMA = """
-- 1. Conversations
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL DEFAULT 'instagram',
    participant_handle TEXT NOT NULL,
    participant_name TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversations_handle ON conversations(participant_handle);

-- 2. Messages
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    fingerprint TEXT UNIQUE NOT NULL,
    sender_type TEXT NOT NULL,
    sender_handle TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'RECEIVED',
    retry_count INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_messages_fingerprint ON messages(fingerprint);
CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);

-- 3. Memories
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(conversation_id, memory_type);

-- 4. Relationship States
CREATE TABLE IF NOT EXISTS relationship_states (
    conversation_id TEXT PRIMARY KEY,
    familiarity REAL NOT NULL DEFAULT 0.1,
    playfulness REAL NOT NULL DEFAULT 0.5,
    sarcasm_tolerance REAL NOT NULL DEFAULT 0.3,
    trust REAL NOT NULL DEFAULT 0.2,
    running_jokes_json TEXT DEFAULT '[]',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- 5. Style Profiles
CREATE TABLE IF NOT EXISTS style_profiles (
    profile_id TEXT PRIMARY KEY,
    conversation_id TEXT UNIQUE NOT NULL,
    avg_sentence_length REAL DEFAULT 8.0,
    formality_score REAL DEFAULT 0.2,
    emoji_density REAL DEFAULT 0.1,
    slang_vocabulary_json TEXT DEFAULT '{}',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- 6. Style Observations
CREATE TABLE IF NOT EXISTS style_observations (
    observation_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    trait_name TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.2,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES style_profiles(profile_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_style_obs_lookup ON style_observations(profile_id, trait_name);

-- 7. Documents
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 8. Document Chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);

-- 9. Humor Examples
CREATE TABLE IF NOT EXISTS humor_examples (
    example_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    tone TEXT NOT NULL,
    intensity REAL NOT NULL DEFAULT 0.5,
    context_prompt TEXT NOT NULL,
    exemplar_response TEXT NOT NULL,
    tags_json TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_humor_cat ON humor_examples(category, intensity);

-- 10. Processing State
CREATE TABLE IF NOT EXISTS processing_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 11. Audit Events
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    correlation_id TEXT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    component TEXT NOT NULL,
    payload_json TEXT DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type, severity);

-- 12. Routine Templates
CREATE TABLE IF NOT EXISTS routine_templates (
    template_id TEXT PRIMARY KEY,
    day_of_week TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    activity TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT 'home',
    default_availability TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routine_day ON routine_templates(day_of_week, start_time);

-- 13. Availability State
CREATE TABLE IF NOT EXISTS availability_state (
    character_id TEXT PRIMARY KEY,
    current_state TEXT NOT NULL,
    current_activity TEXT NOT NULL,
    current_location TEXT NOT NULL,
    away_until TIMESTAMP,
    warning_sent_for_activity TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 14. Academic Subjects & Exams
CREATE TABLE IF NOT EXISTS academic_subjects (
    subject_id TEXT PRIMARY KEY,
    subject_name TEXT NOT NULL,
    current_topic TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS academic_exams (
    exam_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    exam_date TEXT NOT NULL,
    importance TEXT NOT NULL DEFAULT 'HIGH',
    status TEXT NOT NULL DEFAULT 'SCHEDULED',
    result_notes TEXT,
    FOREIGN KEY (subject_id) REFERENCES academic_subjects(subject_id) ON DELETE CASCADE
);

-- 15. Homework Tasks
CREATE TABLE IF NOT EXISTS homework_tasks (
    task_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    description TEXT NOT NULL,
    due_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    priority TEXT NOT NULL DEFAULT 'MEDIUM',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES academic_subjects(subject_id) ON DELETE CASCADE
);

-- 16. Special Events & Life Events
CREATE TABLE IF NOT EXISTS special_events (
    event_id TEXT PRIMARY KEY,
    event_date TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    title TEXT NOT NULL,
    override_activity TEXT NOT NULL,
    override_availability TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS life_events (
    event_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    headline TEXT NOT NULL,
    sentiment TEXT NOT NULL DEFAULT 'NEUTRAL',
    emotional_impact_json TEXT DEFAULT '{}',
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    mentioned_in_chat INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_life_events_time ON life_events(occurred_at DESC);

-- 17. Historical Conversations & Messages
CREATE TABLE IF NOT EXISTS historical_conversations (
    conversation_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    contact_display_name TEXT NOT NULL,
    operator_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    date_start TEXT,
    date_end TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hist_conv_contact ON historical_conversations(contact_id);

CREATE TABLE IF NOT EXISTS historical_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    contact_id TEXT NOT NULL,
    sender_name TEXT NOT NULL,
    sender_is_operator INTEGER NOT NULL DEFAULT 0,
    timestamp_ms INTEGER NOT NULL,
    timestamp_iso TEXT NOT NULL,
    message_type TEXT NOT NULL,
    text TEXT NOT NULL,
    shared_url TEXT,
    share_text TEXT,
    original_content_owner TEXT,
    reactions_json TEXT DEFAULT '[]',
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES historical_conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_hist_msg_conv ON historical_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_hist_msg_contact ON historical_messages(contact_id, timestamp_ms ASC);
CREATE INDEX IF NOT EXISTS idx_hist_msg_type ON historical_messages(message_type);

-- 18. Operator Style Profiles & Observations
CREATE TABLE IF NOT EXISTS operator_style_profiles (
    version_id TEXT PRIMARY KEY,
    total_messages INTEGER NOT NULL DEFAULT 0,
    hinglish_ratio REAL DEFAULT 0.0,
    english_ratio REAL DEFAULT 0.0,
    hindi_ratio REAL DEFAULT 0.0,
    lowercase_ratio REAL DEFAULT 0.0,
    ending_period_ratio REAL DEFAULT 0.0,
    contraction_apostrophe_ratio REAL DEFAULT 0.0,
    avg_message_length_chars REAL DEFAULT 0.0,
    median_message_length_chars REAL DEFAULT 0.0,
    avg_words_per_message REAL DEFAULT 0.0,
    burst_message_ratio REAL DEFAULT 0.0,
    emoji_density REAL DEFAULT 0.0,
    favorite_emojis_json TEXT DEFAULT '[]',
    slang_frequencies_json TEXT DEFAULT '{}',
    question_frequency REAL DEFAULT 0.0,
    common_openers_json TEXT DEFAULT '[]',
    recent_style_json TEXT DEFAULT '{}',
    long_term_style_json TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS operator_style_observations (
    observation_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    trait_name TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

-- 19. Contact Style Profiles & Observations
CREATE TABLE IF NOT EXISTS contact_style_profiles (
    contact_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    total_messages INTEGER NOT NULL DEFAULT 0,
    brevity_level TEXT DEFAULT 'medium',
    avg_message_length_chars REAL DEFAULT 0.0,
    hinglish_ratio REAL DEFAULT 0.0,
    emoji_density REAL DEFAULT 0.0,
    sarcasm_score REAL DEFAULT 0.0,
    formality_score REAL DEFAULT 0.0,
    favorite_emojis_json TEXT DEFAULT '[]',
    slang_terms_json TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contact_style_observations (
    observation_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    trait_name TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

-- 20. Relationship Style Profiles & Observations
CREATE TABLE IF NOT EXISTS relationship_style_profiles (
    contact_id TEXT PRIMARY KEY,
    playfulness REAL DEFAULT 0.5,
    sarcasm REAL DEFAULT 0.3,
    teasing REAL DEFAULT 0.3,
    seriousness REAL DEFAULT 0.4,
    operator_initiation_ratio REAL DEFAULT 0.5,
    operator_message_ratio REAL DEFAULT 0.5,
    avg_exchange_length REAL DEFAULT 4.0,
    reel_sharing_frequency REAL DEFAULT 0.0,
    response_style TEXT DEFAULT 'casual',
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relationship_style_observations (
    observation_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL
);

-- 21. Humor Profiles & Observations
CREATE TABLE IF NOT EXISTS humor_profiles (
    contact_id TEXT PRIMARY KEY,
    humor_frequency REAL DEFAULT 0.0,
    sarcasm_frequency REAL DEFAULT 0.0,
    teasing_frequency REAL DEFAULT 0.0,
    dark_humor_frequency REAL DEFAULT 0.0,
    callback_frequency REAL DEFAULT 0.0,
    laughing_reactions_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS humor_observations (
    observation_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    category TEXT NOT NULL,
    pattern_description TEXT NOT NULL,
    exemplar_text TEXT NOT NULL,
    confidence REAL NOT NULL
);

-- 22. Historical Reels & Interactions
CREATE TABLE IF NOT EXISTS historical_reels (
    reel_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    contact_id TEXT NOT NULL,
    sender_name TEXT NOT NULL,
    sender_is_operator INTEGER NOT NULL DEFAULT 0,
    reel_url TEXT NOT NULL,
    share_text TEXT,
    original_content_owner TEXT,
    timestamp_ms INTEGER NOT NULL,
    reaction_count INTEGER DEFAULT 0,
    topic_category TEXT DEFAULT 'general'
);
CREATE INDEX IF NOT EXISTS idx_hist_reels_url ON historical_reels(reel_url);

CREATE TABLE IF NOT EXISTS reel_interactions (
    interaction_id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL,
    recipient_reacted INTEGER NOT NULL DEFAULT 0,
    reaction_type TEXT,
    follow_up_comment TEXT,
    response_delay_ms INTEGER
);

-- 23. Historical Memories & Topics
CREATE TABLE IF NOT EXISTS historical_memories (
    memory_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    category TEXT NOT NULL,
    statement TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    frequency INTEGER NOT NULL DEFAULT 1,
    recency TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_hist_mem_contact ON historical_memories(contact_id, importance DESC);

CREATE TABLE IF NOT EXISTS historical_topics (
    topic_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    last_mentioned_at TEXT NOT NULL
);

-- 24. Profile Versions & Learning Runs
CREATE TABLE IF NOT EXISTS style_profile_versions (
    version_id TEXT PRIMARY KEY,
    dataset_hash TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    conversation_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS learning_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    dataset_hash TEXT NOT NULL,
    files_processed INTEGER NOT NULL DEFAULT 0,
    messages_processed INTEGER NOT NULL DEFAULT 0,
    observations_created INTEGER NOT NULL DEFAULT 0,
    memories_created INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'RUNNING'
);

CREATE TABLE IF NOT EXISTS learning_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 25. Historical Contextual Turns & Behavioral Patterns (Conversation Intelligence v2)
CREATE TABLE IF NOT EXISTS historical_turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    contact_id TEXT NOT NULL,
    contact_text TEXT NOT NULL,
    operator_text TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    social_act TEXT NOT NULL DEFAULT 'other',
    social_intent_json TEXT DEFAULT '{}',
    response_strategy TEXT NOT NULL DEFAULT 'direct_answer',
    response_length_category TEXT NOT NULL DEFAULT 'short',
    context_messages_json TEXT DEFAULT '[]',
    turn_energy REAL DEFAULT 0.5,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hist_turns_contact ON historical_turns(contact_id);
CREATE INDEX IF NOT EXISTS idx_hist_turns_act ON historical_turns(social_act, response_strategy);

CREATE TABLE IF NOT EXISTS behavioral_patterns (
    pattern_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL DEFAULT 'global',
    trigger_social_act TEXT NOT NULL,
    response_strategy TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.5,
    typical_length TEXT NOT NULL DEFAULT 'short',
    example_pairs_json TEXT DEFAULT '[]',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_patterns_trigger ON behavioral_patterns(trigger_social_act, contact_id);
"""


class DatabaseManager:
    """Manages SQLite database connections and PRAGMA settings."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            settings = get_settings()
            self.db_path = settings.resolved_database_path
        else:
            self.db_path = Path(db_path)

        self._lock = asyncio.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection with row factories and PRAGMAs."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode by default, explicit transactions managed manually
        )
        conn.row_factory = sqlite3.Row

        # Apply production PRAGMAs
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.close()

        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager providing an explicit transaction with automatic commit and rollback."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE;")
            yield cursor
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise
        finally:
            cursor.close()
            conn.close()

    def initialize_schema(self) -> None:
        """Run initial database migrations and create all tables and indices."""
        logger.info("database.initializing_schema", db_path=str(self.db_path))
        try:
            conn = self.get_connection()
            try:
                conn.executescript(DDL_SCHEMA)
                logger.info("database.schema_initialized_successfully")
            finally:
                conn.close()
        except Exception as e:
            logger.error("database.initialization_failed", error=str(e))
            raise DatabaseException(f"Failed to execute schema initialization: {e}") from e


# Global singleton instance
_db_manager: DatabaseManager | None = None


def get_db_manager(db_path: Path | str | None = None) -> DatabaseManager:
    """Get or create singleton DatabaseManager."""
    global _db_manager
    if _db_manager is None or db_path is not None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager
