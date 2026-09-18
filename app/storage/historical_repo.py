"""Repository for historical conversations, style profiles, humor, reels, and learning runs."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.core.logging import get_logger
from app.learning.models import (
    ContactStyleProfile,
    HumorProfile,
    MemoryCandidate,
    NormalizedConversation,
    NormalizedMessage,
    OperatorStyleProfile,
    ReelIntelligenceProfile,
    RelationshipStyleProfile,
    HistoricalTurn,
    BehavioralPattern,
)
from app.storage.database import DatabaseManager

logger = get_logger("storage.historical")


class HistoricalRepository:
    """Manages persistence and retrieval for historical dataset and learned style profiles."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.db.initialize_schema()

    def upsert_conversation(
        self,
        conversation: NormalizedConversation,
        contact_id: str,
        contact_name: str,
        operator_name: str,
    ) -> None:
        """Upsert a historical conversation record."""
        query = """
        INSERT INTO historical_conversations (
            conversation_id, contact_id, contact_display_name, operator_name,
            file_path, message_count, date_start, date_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            contact_id=excluded.contact_id,
            contact_display_name=excluded.contact_display_name,
            operator_name=excluded.operator_name,
            file_path=excluded.file_path,
            message_count=excluded.message_count,
            date_start=excluded.date_start,
            date_end=excluded.date_end
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    conversation.conversation_id,
                    contact_id,
                    contact_name,
                    operator_name,
                    conversation.file_path,
                    conversation.message_count,
                    conversation.date_start,
                    conversation.date_end,
                ),
            )

    def batch_insert_messages(
        self,
        messages: list[NormalizedMessage],
        contact_id: str,
        operator_name: str,
    ) -> int:
        """Batch insert normalized messages into historical_messages."""
        if not messages:
            return 0

        query = """
        INSERT OR IGNORE INTO historical_messages (
            message_id, conversation_id, contact_id, sender_name,
            sender_is_operator, timestamp_ms, timestamp_iso, message_type,
            text, shared_url, share_text, original_content_owner,
            reactions_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        records: list[tuple[Any, ...]] = []
        op_lower = operator_name.strip().lower()

        for idx, m in enumerate(messages):
            mid = f"{m.conversation_id}_{m.timestamp_ms}_{idx}"
            is_op = 1 if m.sender_display_name.strip().lower() == op_lower else 0
            reactions_json = json.dumps([r.model_dump() for r in m.reactions])
            metadata_json = json.dumps(m.metadata)

            records.append((
                mid,
                m.conversation_id,
                contact_id,
                m.sender_display_name,
                is_op,
                m.timestamp_ms,
                m.timestamp_iso,
                m.message_type.value,
                m.text,
                m.shared_url,
                m.share_text,
                m.original_content_owner,
                reactions_json,
                metadata_json,
            ))

        with self.db.transaction() as cur:
            cur.executemany(query, records)

        return len(records)

    def get_conversations(self) -> list[dict[str, Any]]:
        """Retrieve all historical conversations metadata."""
        query = "SELECT * FROM historical_conversations ORDER BY date_start ASC"
        with self.db.transaction() as cur:
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]

    def get_messages_for_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a specific conversation in chronological order."""
        query = "SELECT * FROM historical_messages WHERE conversation_id = ? ORDER BY timestamp_ms ASC"
        with self.db.transaction() as cur:
            cur.execute(query, (conversation_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_all_operator_messages(self) -> list[dict[str, Any]]:
        """Retrieve all historical messages sent by the operator."""
        query = "SELECT * FROM historical_messages WHERE sender_is_operator = 1 ORDER BY timestamp_ms ASC"
        with self.db.transaction() as cur:
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]

    def get_messages_for_contact(self, contact_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages exchanged in conversations with a specific contact."""
        query = "SELECT * FROM historical_messages WHERE contact_id = ? ORDER BY timestamp_ms ASC"
        with self.db.transaction() as cur:
            cur.execute(query, (contact_id,))
            return [dict(r) for r in cur.fetchall()]

    def upsert_operator_style(self, profile: OperatorStyleProfile) -> None:
        """Upsert global learned operator style profile."""
        query = """
        INSERT INTO operator_style_profiles (
            version_id, total_messages, hinglish_ratio, english_ratio, hindi_ratio,
            lowercase_ratio, ending_period_ratio, contraction_apostrophe_ratio,
            avg_message_length_chars, median_message_length_chars, avg_words_per_message,
            burst_message_ratio, emoji_density, favorite_emojis_json, slang_frequencies_json,
            question_frequency, common_openers_json, recent_style_json, long_term_style_json,
            confidence, evidence_count, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(version_id) DO UPDATE SET
            total_messages=excluded.total_messages,
            hinglish_ratio=excluded.hinglish_ratio,
            english_ratio=excluded.english_ratio,
            hindi_ratio=excluded.hindi_ratio,
            lowercase_ratio=excluded.lowercase_ratio,
            ending_period_ratio=excluded.ending_period_ratio,
            contraction_apostrophe_ratio=excluded.contraction_apostrophe_ratio,
            avg_message_length_chars=excluded.avg_message_length_chars,
            median_message_length_chars=excluded.median_message_length_chars,
            avg_words_per_message=excluded.avg_words_per_message,
            burst_message_ratio=excluded.burst_message_ratio,
            emoji_density=excluded.emoji_density,
            favorite_emojis_json=excluded.favorite_emojis_json,
            slang_frequencies_json=excluded.slang_frequencies_json,
            question_frequency=excluded.question_frequency,
            common_openers_json=excluded.common_openers_json,
            recent_style_json=excluded.recent_style_json,
            long_term_style_json=excluded.long_term_style_json,
            confidence=excluded.confidence,
            evidence_count=excluded.evidence_count,
            updated_at=CURRENT_TIMESTAMP
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    profile.version_id,
                    profile.total_messages,
                    profile.hinglish_ratio,
                    profile.english_ratio,
                    profile.hindi_ratio,
                    profile.lowercase_ratio,
                    profile.ending_period_ratio,
                    profile.contraction_apostrophe_ratio,
                    profile.avg_message_length_chars,
                    profile.median_message_length_chars,
                    profile.avg_words_per_message,
                    profile.burst_message_ratio,
                    profile.emoji_density,
                    json.dumps(profile.favorite_emojis),
                    json.dumps(profile.slang_frequencies),
                    profile.question_frequency,
                    json.dumps(profile.common_openers),
                    json.dumps(profile.recent_style),
                    json.dumps(profile.long_term_style),
                    profile.confidence,
                    profile.evidence_count,
                ),
            )

    def get_operator_style(self, version_id: str | None = None) -> dict[str, Any] | None:
        """Retrieve operator style profile."""
        if version_id:
            query = "SELECT * FROM operator_style_profiles WHERE version_id = ?"
            params: tuple[Any, ...] = (version_id,)
        else:
            query = """
            SELECT osp.* FROM operator_style_profiles osp
            LEFT JOIN style_profile_versions spv ON osp.version_id = spv.version_id
            ORDER BY COALESCE(spv.is_active, 0) DESC, osp.updated_at DESC
            LIMIT 1
            """
            params = ()
        with self.db.transaction() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def upsert_contact_style(self, profile: ContactStyleProfile) -> None:
        """Upsert style profile for an individual contact."""
        query = """
        INSERT INTO contact_style_profiles (
            contact_id, display_name, total_messages, brevity_level,
            avg_message_length_chars, hinglish_ratio, emoji_density,
            sarcasm_score, formality_score, favorite_emojis_json,
            slang_terms_json, confidence, evidence_count, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(contact_id) DO UPDATE SET
            display_name=excluded.display_name,
            total_messages=excluded.total_messages,
            brevity_level=excluded.brevity_level,
            avg_message_length_chars=excluded.avg_message_length_chars,
            hinglish_ratio=excluded.hinglish_ratio,
            emoji_density=excluded.emoji_density,
            sarcasm_score=excluded.sarcasm_score,
            formality_score=excluded.formality_score,
            favorite_emojis_json=excluded.favorite_emojis_json,
            slang_terms_json=excluded.slang_terms_json,
            confidence=excluded.confidence,
            evidence_count=excluded.evidence_count,
            updated_at=CURRENT_TIMESTAMP
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    profile.contact_id,
                    profile.display_name,
                    profile.total_messages,
                    profile.brevity_level,
                    profile.avg_message_length_chars,
                    profile.hinglish_ratio,
                    profile.emoji_density,
                    profile.sarcasm_score,
                    profile.formality_score,
                    json.dumps(profile.favorite_emojis),
                    json.dumps(profile.slang_terms_used),
                    profile.confidence,
                    profile.evidence_count,
                ),
            )

    def get_contact_style(self, contact_id: str) -> dict[str, Any] | None:
        """Retrieve style profile for a contact."""
        query = "SELECT * FROM contact_style_profiles WHERE contact_id = ?"
        with self.db.transaction() as cur:
            cur.execute(query, (contact_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_all_contact_styles(self) -> list[dict[str, Any]]:
        """Retrieve all contact style profiles."""
        query = "SELECT * FROM contact_style_profiles ORDER BY total_messages DESC"
        with self.db.transaction() as cur:
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]

    def upsert_relationship_style(self, profile: RelationshipStyleProfile) -> None:
        """Upsert dyadic relationship profile."""
        query = """
        INSERT INTO relationship_style_profiles (
            contact_id, playfulness, sarcasm, teasing, seriousness,
            operator_initiation_ratio, operator_message_ratio, avg_exchange_length,
            reel_sharing_frequency, response_style, confidence, evidence_count, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(contact_id) DO UPDATE SET
            playfulness=excluded.playfulness,
            sarcasm=excluded.sarcasm,
            teasing=excluded.teasing,
            seriousness=excluded.seriousness,
            operator_initiation_ratio=excluded.operator_initiation_ratio,
            operator_message_ratio=excluded.operator_message_ratio,
            avg_exchange_length=excluded.avg_exchange_length,
            reel_sharing_frequency=excluded.reel_sharing_frequency,
            response_style=excluded.response_style,
            confidence=excluded.confidence,
            evidence_count=excluded.evidence_count,
            updated_at=CURRENT_TIMESTAMP
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    profile.contact_id,
                    profile.playfulness,
                    profile.sarcasm,
                    profile.teasing,
                    profile.seriousness,
                    profile.operator_initiation_ratio,
                    profile.operator_message_ratio,
                    profile.avg_exchange_length,
                    profile.reel_sharing_frequency,
                    profile.response_style,
                    profile.confidence,
                    profile.evidence_count,
                ),
            )

    def get_relationship_style(self, contact_id: str) -> dict[str, Any] | None:
        """Retrieve relationship profile for a contact."""
        query = "SELECT * FROM relationship_style_profiles WHERE contact_id = ?"
        with self.db.transaction() as cur:
            cur.execute(query, (contact_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def upsert_humor_profile(self, profile: HumorProfile) -> None:
        """Upsert humor profile."""
        query = """
        INSERT INTO humor_profiles (
            contact_id, humor_frequency, sarcasm_frequency, teasing_frequency,
            dark_humor_frequency, callback_frequency, laughing_reactions_count,
            confidence, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(contact_id) DO UPDATE SET
            humor_frequency=excluded.humor_frequency,
            sarcasm_frequency=excluded.sarcasm_frequency,
            teasing_frequency=excluded.teasing_frequency,
            dark_humor_frequency=excluded.dark_humor_frequency,
            callback_frequency=excluded.callback_frequency,
            laughing_reactions_count=excluded.laughing_reactions_count,
            confidence=excluded.confidence,
            updated_at=CURRENT_TIMESTAMP
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    profile.contact_id,
                    profile.humor_frequency,
                    profile.sarcasm_frequency,
                    profile.teasing_frequency,
                    profile.dark_humor_frequency,
                    profile.callback_frequency,
                    profile.laughing_reactions_count,
                    profile.confidence,
                ),
            )

    def get_humor_profile(self, contact_id: str = "global") -> dict[str, Any] | None:
        """Retrieve humor profile."""
        query = "SELECT * FROM humor_profiles WHERE contact_id = ?"
        with self.db.transaction() as cur:
            cur.execute(query, (contact_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def insert_reels(self, reels: list[dict[str, Any]]) -> None:
        """Batch insert historical reel shares."""
        if not reels:
            return
        query = """
        INSERT OR IGNORE INTO historical_reels (
            reel_id, conversation_id, contact_id, sender_name,
            sender_is_operator, reel_url, share_text, original_content_owner,
            timestamp_ms, reaction_count, topic_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        records = [
            (
                r["reel_id"],
                r["conversation_id"],
                r["contact_id"],
                r["sender_name"],
                r["sender_is_operator"],
                r["reel_url"],
                r.get("share_text"),
                r.get("original_content_owner"),
                r["timestamp_ms"],
                r.get("reaction_count", 0),
                r.get("topic_category", "general"),
            )
            for r in reels
        ]
        with self.db.transaction() as cur:
            cur.executemany(query, records)

    def get_all_reels(self) -> list[dict[str, Any]]:
        """Retrieve all historical reels."""
        query = "SELECT * FROM historical_reels ORDER BY timestamp_ms DESC"
        with self.db.transaction() as cur:
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]

    def is_reel_previously_shared(self, reel_url: str) -> bool:
        """Check if reel URL exists in historical or sent reels."""
        query = "SELECT 1 FROM historical_reels WHERE reel_url = ? LIMIT 1"
        with self.db.transaction() as cur:
            cur.execute(query, (reel_url,))
            return cur.fetchone() is not None

    def upsert_memory_candidate(self, mem: MemoryCandidate) -> None:
        """Upsert extracted candidate memory."""
        query = """
        INSERT INTO historical_memories (
            memory_id, contact_id, category, statement,
            importance, confidence, frequency, recency, source_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            importance=excluded.importance,
            confidence=excluded.confidence,
            frequency=excluded.frequency,
            recency=excluded.recency,
            source_count=excluded.source_count
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    mem.memory_id,
                    mem.contact_id,
                    mem.category,
                    mem.statement,
                    mem.importance,
                    mem.confidence,
                    mem.frequency,
                    mem.recency,
                    mem.source_count,
                ),
            )

    def get_top_memories(self, contact_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve top memory candidates by importance and confidence."""
        if contact_id:
            query = """
            SELECT * FROM historical_memories
            WHERE contact_id = ? AND confidence >= 0.6
            ORDER BY importance DESC, confidence DESC LIMIT ?
            """
            params: tuple[Any, ...] = (contact_id, limit)
        else:
            query = """
            SELECT * FROM historical_memories
            WHERE confidence >= 0.6
            ORDER BY importance DESC, confidence DESC LIMIT ?
            """
            params = (limit,)

        with self.db.transaction() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def create_version(
        self,
        version_id: str,
        dataset_hash: str,
        algorithm_version: str,
        message_count: int,
        conversation_count: int,
    ) -> None:
        """Record a newly built style profile version and mark it active."""
        with self.db.transaction() as cur:
            cur.execute("UPDATE style_profile_versions SET is_active = 0")
            cur.execute(
                """
                INSERT INTO style_profile_versions (
                    version_id, dataset_hash, algorithm_version, message_count,
                    conversation_count, is_active
                ) VALUES (?, ?, ?, ?, ?, 1)
                """,
                (version_id, dataset_hash, algorithm_version, message_count, conversation_count),
            )

    def get_active_version(self) -> dict[str, Any] | None:
        """Retrieve currently active profile version record."""
        query = "SELECT * FROM style_profile_versions WHERE is_active = 1 LIMIT 1"
        with self.db.transaction() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return dict(row) if row else None

    def record_learning_run(
        self,
        run_id: str,
        dataset_hash: str,
        files_processed: int,
        messages_processed: int,
        observations_created: int,
        memories_created: int,
        status: str = "COMPLETED",
        errors: list[str] | None = None,
    ) -> None:
        """Record an executed learning run."""
        query = """
        INSERT INTO learning_runs (
            run_id, completed_at, dataset_hash, files_processed, messages_processed,
            observations_created, memories_created, errors_json, status
        ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as cur:
            cur.execute(
                query,
                (
                    run_id,
                    dataset_hash,
                    files_processed,
                    messages_processed,
                    observations_created,
                    memories_created,
                    json.dumps(errors or []),
                    status,
                ),
            )

    def batch_insert_turns(self, turns: list[HistoricalTurn]) -> int:
        """Batch insert contextual turn examples."""
        if not turns:
            return 0
        query = """
        INSERT OR REPLACE INTO historical_turns (
            turn_id, conversation_id, contact_id, contact_text, operator_text,
            timestamp_ms, social_act, social_intent_json, response_strategy,
            response_length_category, context_messages_json, turn_energy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        records = [
            (
                t.turn_id,
                t.conversation_id,
                t.contact_id,
                t.contact_text,
                t.operator_text,
                t.timestamp_ms,
                t.social_act,
                json.dumps(t.social_intent),
                t.response_strategy,
                t.response_length_category,
                json.dumps(t.context_messages),
                t.turn_energy,
            )
            for t in turns
        ]
        with self.db.transaction() as cur:
            cur.executemany(query, records)
        return len(records)

    def get_turns(
        self,
        contact_id: str | None = None,
        social_act: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve historical turn examples filtered by contact or social act."""
        clauses = []
        params: list[Any] = []
        if contact_id:
            clauses.append("contact_id = ?")
            params.append(contact_id)
        if social_act:
            clauses.append("social_act = ?")
            params.append(social_act)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
        SELECT turn_id, conversation_id, contact_id, contact_text, operator_text,
               timestamp_ms, social_act, social_intent_json, response_strategy,
               response_length_category, context_messages_json, turn_energy
        FROM historical_turns
        {where_sql}
        ORDER BY timestamp_ms DESC
        LIMIT ?
        """
        params.append(limit)
        with self.db.transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {
                    "turn_id": r[0],
                    "conversation_id": r[1],
                    "contact_id": r[2],
                    "contact_text": r[3],
                    "operator_text": r[4],
                    "timestamp_ms": r[5],
                    "social_act": r[6],
                    "social_intent": json.loads(r[7]) if r[7] else {},
                    "response_strategy": r[8],
                    "response_length_category": r[9],
                    "context_messages": json.loads(r[10]) if r[10] else [],
                    "turn_energy": r[11],
                }
                for r in rows
            ]

    def upsert_behavioral_patterns(self, patterns: list[BehavioralPattern]) -> int:
        """Upsert statistical behavioral patterns."""
        if not patterns:
            return 0
        query = """
        INSERT INTO behavioral_patterns (
            pattern_id, contact_id, trigger_social_act, response_strategy,
            sample_count, confidence, typical_length, example_pairs_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pattern_id) DO UPDATE SET
            sample_count=excluded.sample_count,
            confidence=excluded.confidence,
            typical_length=excluded.typical_length,
            example_pairs_json=excluded.example_pairs_json,
            updated_at=CURRENT_TIMESTAMP
        """
        records = [
            (
                p.pattern_id,
                p.contact_id,
                p.trigger_social_act,
                p.response_strategy,
                p.sample_count,
                p.confidence,
                p.typical_length,
                json.dumps(p.example_pairs),
            )
            for p in patterns
        ]
        with self.db.transaction() as cur:
            cur.executemany(query, records)
        return len(records)

    def get_behavioral_patterns(
        self,
        contact_id: str | None = None,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Retrieve learned behavioral patterns, falling back to global patterns."""
        query = """
        SELECT pattern_id, contact_id, trigger_social_act, response_strategy,
               sample_count, confidence, typical_length, example_pairs_json
        FROM behavioral_patterns
        WHERE contact_id = ? OR contact_id = 'global'
        ORDER BY confidence DESC, sample_count DESC
        LIMIT ?
        """
        with self.db.transaction() as cur:
            cur.execute(query, (contact_id or "global", limit))
            rows = cur.fetchall()
            return [
                {
                    "pattern_id": r[0],
                    "contact_id": r[1],
                    "trigger_social_act": r[2],
                    "response_strategy": r[3],
                    "sample_count": r[4],
                    "confidence": r[5],
                    "typical_length": r[6],
                    "example_pairs": json.loads(r[7]) if r[7] else [],
                }
                for r in rows
            ]

    def get_pattern_for_act(
        self,
        trigger_social_act: str,
        contact_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get best matching behavioral pattern for a specific speech act."""
        query = """
        SELECT pattern_id, contact_id, trigger_social_act, response_strategy,
               sample_count, confidence, typical_length, example_pairs_json
        FROM behavioral_patterns
        WHERE trigger_social_act = ? AND (contact_id = ? OR contact_id = 'global')
        ORDER BY (contact_id = ?) DESC, confidence DESC
        LIMIT 1
        """
        cid = contact_id or "global"
        with self.db.transaction() as cur:
            cur.execute(query, (trigger_social_act, cid, cid))
            r = cur.fetchone()
            if not r:
                return None
            return {
                "pattern_id": r[0],
                "contact_id": r[1],
                "trigger_social_act": r[2],
                "response_strategy": r[3],
                "sample_count": r[4],
                "confidence": r[5],
                "typical_length": r[6],
                "example_pairs": json.loads(r[7]) if r[7] else [],
            }

    def reset_derived_profiles(self) -> None:
        """Reset derived profiles, observations, humor, memories, turns, and patterns."""
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM operator_style_profiles")
            cur.execute("DELETE FROM operator_style_observations")
            cur.execute("DELETE FROM contact_style_profiles")
            cur.execute("DELETE FROM contact_style_observations")
            cur.execute("DELETE FROM relationship_style_profiles")
            cur.execute("DELETE FROM relationship_style_observations")
            cur.execute("DELETE FROM humor_profiles")
            cur.execute("DELETE FROM humor_observations")
            cur.execute("DELETE FROM historical_memories")
            cur.execute("DELETE FROM historical_topics")
            cur.execute("DELETE FROM historical_turns")
            cur.execute("DELETE FROM behavioral_patterns")
            cur.execute("DELETE FROM style_profile_versions")
        logger.info("historical_repo.derived_profiles_reset")

    def full_purge(self) -> None:
        """Purge all historical tables completely."""
        self.reset_derived_profiles()
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM historical_reels")
            cur.execute("DELETE FROM reel_interactions")
            cur.execute("DELETE FROM historical_messages")
            cur.execute("DELETE FROM historical_conversations")
            cur.execute("DELETE FROM learning_runs")
            cur.execute("DELETE FROM learning_observations")
        logger.info("historical_repo.full_purge_completed")
