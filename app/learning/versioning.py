"""Style profile versioning, learning run orchestration, and rebuild coordinator."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.learning.contact_style import ContactStyleAnalyzer
from app.learning.dataset_scanner import DatasetScanner
from app.learning.humor_learning import HumorAnalyzer
from app.learning.memory_learning import MemoryExtractor
from app.learning.models import BehavioralPattern
from app.learning.normalizer import InstagramNormalizer
from app.learning.operator_detector import OperatorDetector
from app.learning.operator_style import OperatorStyleAnalyzer
from app.learning.rag_indexer import HistoricalRAGIndexer
from app.learning.reel_intelligence import ReelIntelligenceAnalyzer
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.historical_repo import HistoricalRepository

from dataclasses import dataclass, field

logger = get_logger("learning.versioning")

ALGORITHM_VERSION = "2.0.0-historical-intelligence"


@dataclass
class ImportSummary:
    files_discovered: int = 0
    files_processed: int = 0
    files_rejected: dict[str, str] = field(default_factory=dict)
    messages_imported: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)


class LearningCoordinator:
    """Orchestrates end-to-end historical learning runs, rebuilds, and profile versioning."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.repo = HistoricalRepository(self.db)
        self.settings = get_settings()
        self.last_import_summary = ImportSummary()

    def import_dataset(
        self,
        dataset_path: Path | str | None = None,
        limit: int | None = None,
    ) -> tuple[int, int]:
        """Scan dataset folder, normalize all conversations, and persist to SQLite."""
        scanner = DatasetScanner(dataset_path)
        manifest = scanner.scan()
        scanner.save_manifest(manifest)

        detector = OperatorDetector()
        target_dir = Path(dataset_path or self.settings.resolved_historical_dataset_path)
        files = list(target_dir.rglob("*.json"))
        candidate_files = [f for f in files if "manifest" not in f.name and "processed" not in str(f)]

        if limit:
            candidate_files = candidate_files[:limit]

        summary = ImportSummary(
            files_discovered=manifest.files_discovered,
            files_rejected=dict(manifest.invalid_file_reasons),
        )
        self.last_import_summary = summary

        conv_count = 0
        msg_count = 0

        for f in candidate_files:
            try:
                norm_conv = InstagramNormalizer.normalize_conversation(
                    file_path=f,
                    operator_names=self.settings.OPERATOR_NAMES,
                )
            except Exception as e:
                summary.files_rejected[f.name] = f"Normalization failed: {e}"
                logger.warning("learning.file_normalization_skipped", file=f.name, error=str(e))
                continue

            # Identify operator and contacts
            try:
                op_name, contacts = detector.identify_participants(
                    norm_conv.participants, conversation_file=f.name
                )
            except Exception as e:
                summary.files_rejected[f.name] = f"Operator identification failed: {e}"
                logger.error("learning.operator_identification_failed", file=f.name, error=str(e))
                raise

            contact_name = contacts[0] if contacts else "DirectUser"
            contact_id = detector.generate_contact_id(contact_name)

            self.repo.upsert_conversation(
                conversation=norm_conv,
                contact_id=contact_id,
                contact_name=contact_name,
                operator_name=op_name,
            )
            inserted_msgs = self.repo.batch_insert_messages(
                messages=norm_conv.messages,
                contact_id=contact_id,
                operator_name=op_name,
            )

            # Extract and persist contextual turns
            turns = InstagramNormalizer.extract_contextual_turns(
                norm_conv,
                operator_name=op_name,
                contact_id=contact_id,
            )
            if turns:
                self.repo.batch_insert_turns(turns)

            conv_count += 1
            msg_count += inserted_msgs
            summary.files_processed += 1
            summary.messages_imported += inserted_msgs

            for m in norm_conv.messages:
                t = m.message_type.value
                summary.type_counts[t] = summary.type_counts.get(t, 0) + 1

        logger.info("learning.dataset_imported", conversations=conv_count, messages=msg_count)
        return conv_count, msg_count

    def run_analysis(self, version_id: str | None = None, index_rag: bool = True) -> str:
        """Analyze imported historical data and generate versioned style profiles."""
        t0 = time.time()
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        conversations = self.repo.get_conversations()
        operator_name = self.settings.OPERATOR_NAMES[0] if self.settings.OPERATOR_NAMES else "Arnav Srivastava"

        # Determine next version ID
        if not version_id:
            active = self.repo.get_active_version()
            if active:
                old_ver = active.get("version_id", "v000")
                try:
                    num = int(old_ver.replace("v", "").replace("style_profile_", ""))
                    version_id = f"style_profile_v{num + 1:03d}"
                except ValueError:
                    version_id = f"style_profile_v{int(time.time())}"
            else:
                version_id = "style_profile_v001"

        logger.info("learning.analysis_started", run_id=run_id, version=version_id)

        # 1. Global Operator Style
        op_analyzer = OperatorStyleAnalyzer(
            recent_window_days=self.settings.HISTORICAL_RECENT_WINDOW_DAYS,
            long_term_weight=self.settings.HISTORICAL_LONG_TERM_WEIGHT,
            recent_weight=self.settings.HISTORICAL_RECENT_WEIGHT,
        )
        all_op_msgs = self.repo.get_all_operator_messages()
        op_profile = op_analyzer.analyze(all_op_msgs, version_id=version_id)
        self.repo.upsert_operator_style(op_profile)

        # 2. Contact Profiles & Dyadic Relationships
        contact_analyzer = ContactStyleAnalyzer()
        from app.learning.relationship_style import RelationshipStyleAnalyzer
        rel_analyzer = RelationshipStyleAnalyzer()
        humor_analyzer = HumorAnalyzer()
        mem_extractor = MemoryExtractor()
        reel_analyzer = ReelIntelligenceAnalyzer()

        # Group all messages by contact
        contacts_seen: dict[str, str] = {}
        for c in conversations:
            cid = c.get("contact_id")
            cname = c.get("contact_display_name", "User")
            if cid:
                contacts_seen[cid] = cname

        all_extracted_reels: list[dict[str, Any]] = []
        total_memories_created = 0

        for contact_id, display_name in contacts_seen.items():
            contact_msgs = self.repo.get_messages_for_contact(contact_id)
            c_msgs = [m for m in contact_msgs if not m.get("sender_is_operator")]

            # Contact style
            c_profile = contact_analyzer.analyze(contact_id, display_name, c_msgs)
            self.repo.upsert_contact_style(c_profile)

            # Relationship style
            rel_profile = rel_analyzer.analyze(contact_id, contact_msgs, operator_name=operator_name)
            self.repo.upsert_relationship_style(rel_profile)

            # Humor profile
            h_profile = humor_analyzer.analyze(contact_id, contact_msgs)
            self.repo.upsert_humor_profile(h_profile)

            # Memory candidates
            mem_candidates, _ = mem_extractor.extract(contact_id, contact_msgs)
            for mem in mem_candidates:
                self.repo.upsert_memory_candidate(mem)
                total_memories_created += 1

            # Reels
            _, reels = reel_analyzer.analyze(contact_msgs, operator_name=operator_name)
            all_extracted_reels.extend(reels)

        # Global Humor Profile
        global_msgs = []
        for contact_id in contacts_seen.keys():
            global_msgs.extend(self.repo.get_messages_for_contact(contact_id))
        global_humor = humor_analyzer.analyze("global", global_msgs)
        self.repo.upsert_humor_profile(global_humor)

        # Store historical reels
        self.repo.insert_reels(all_extracted_reels)

        # 3. Local RAG indexing
        if index_rag and global_msgs:
            try:
                rag_indexer = HistoricalRAGIndexer(db=self.db)
                rag_indexer.index_conversations(conversations, global_msgs)
            except Exception as e:
                logger.warning("learning.rag_indexing_error", error=str(e))
        # 3b. Derive and store Behavioral Patterns across all turns
        all_turns = self.repo.get_turns(limit=5000)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for t in all_turns:
            c_id = t.get("contact_id", "global")
            act = t.get("social_act", "other")
            strat = t.get("response_strategy", "direct_answer")
            groups[(c_id, act, strat)].append(t)
            groups[("global", act, strat)].append(t)

        behavioral_patterns: list[BehavioralPattern] = []
        for (c_id, act, strat), turn_list in groups.items():
            sample_count = len(turn_list)
            lengths = [t.get("response_length_category", "short") for t in turn_list]
            typical_length = Counter(lengths).most_common(1)[0][0] if lengths else "short"
            confidence = min(0.5 + sample_count * 0.04, 0.95)
            example_pairs = [
                {"contact": t.get("contact_text", ""), "operator": t.get("operator_text", "")}
                for t in turn_list[:3]
            ]
            pat_id = f"pat_{c_id}_{act}_{strat}"
            behavioral_patterns.append(
                BehavioralPattern(
                    pattern_id=pat_id,
                    contact_id=c_id,
                    trigger_social_act=act,
                    response_strategy=strat,
                    sample_count=sample_count,
                    confidence=round(confidence, 2),
                    typical_length=typical_length,
                    example_pairs=example_pairs,
                )
            )

        if behavioral_patterns:
            self.repo.upsert_behavioral_patterns(behavioral_patterns)
            logger.info("learning.behavioral_patterns_persisted", count=len(behavioral_patterns))

        # 4. Save Version Record
        scanner = DatasetScanner()
        manifest = scanner.scan()
        total_msgs_count = len(global_msgs)
        self.repo.create_version(
            version_id=version_id,
            dataset_hash=manifest.dataset_hash,
            algorithm_version=ALGORITHM_VERSION,
            message_count=total_msgs_count,
            conversation_count=len(conversations),
        )

        # 5. Record Learning Run
        self.repo.record_learning_run(
            run_id=run_id,
            dataset_hash=manifest.dataset_hash,
            files_processed=manifest.valid_conversations,
            messages_processed=total_msgs_count,
            observations_created=len(contacts_seen) * 4 + 1,
            memories_created=total_memories_created,
            status="COMPLETED",
        )

        duration = round(time.time() - t0, 2)
        logger.info(
            "learning.analysis_completed",
            version=version_id,
            duration_s=duration,
            contacts_profiled=len(contacts_seen),
            memories_extracted=total_memories_created,
        )

        return version_id

    def full_rebuild(self, dataset_path: Path | str | None = None) -> str:
        """Preserve source conversations, wipe derived profiles, re-import, and recalculate everything."""
        logger.info("learning.full_rebuild_initiated")
        self.repo.reset_derived_profiles()
        self.import_dataset(dataset_path=dataset_path)
        version_id = self.run_analysis()
        logger.info("learning.full_rebuild_completed", version=version_id)
        return version_id

    def reset_profiles(self) -> None:
        """Wipe all derived style profiles, observations, humor, and memories."""
        self.repo.reset_derived_profiles()
        logger.info("learning.profiles_reset_completed")
