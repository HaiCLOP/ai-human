"""Conversational chemistry dynamics and relationship state evolution."""

from __future__ import annotations

import re
from typing import Sequence

from app.core.logging import get_logger
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import RelationshipRecord, RelationshipRepository

logger = get_logger("conversation.chemistry")

POSITIVE_HUMOR_MARKERS = [
    re.compile(r"\b(lmao|lmfao|lol|haha|hahaha|rofl|ded|dead)\b", re.IGNORECASE),
    re.compile(r"[😂🤣💀]+"),
]

NEGATIVE_REACTION_MARKERS = [
    re.compile(r"\b(rude|mean|stop\s+it|not\s+funny|unfunny|offensive|apologize|too\s+far)\b", re.IGNORECASE),
]


class ChemistryModel:
    """Modulates continuous behavioral dynamics based on conversational feedback."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.repo = RelationshipRepository(self.db)

    def evaluate_and_update(self, conversation_id: str, user_message: str) -> list[str]:
        """Update chemistry metrics and produce prompt notes."""
        rel = self.repo.get_or_create(conversation_id)

        # 1. Familiarity increases gradually with each turn
        new_fam = min(1.0, round(rel.familiarity + 0.03, 3))

        # 2. Analyze reaction to humor / sarcasm
        has_positive_laugh = any(p.search(user_message) for p in POSITIVE_HUMOR_MARKERS)
        has_negative_recoil = any(p.search(user_message) for p in NEGATIVE_REACTION_MARKERS)

        new_sarcasm = rel.sarcasm_tolerance
        new_playfulness = rel.playfulness

        if has_positive_laugh:
            new_sarcasm = min(1.0, round(new_sarcasm + 0.06, 3))
            new_playfulness = min(1.0, round(new_playfulness + 0.04, 3))
        elif has_negative_recoil:
            new_sarcasm = max(0.1, round(new_sarcasm - 0.20, 3))
            new_playfulness = max(0.2, round(new_playfulness - 0.15, 3))

        # 3. Trust increases slightly when positive banter is sustained
        new_trust = min(1.0, round(rel.trust + (0.02 if has_positive_laugh else 0.005), 3))

        self.repo.update_chemistry(
            conversation_id=conversation_id,
            familiarity=new_fam,
            playfulness=new_playfulness,
            sarcasm_tolerance=new_sarcasm,
            trust=new_trust,
        )

        logger.info(
            "chemistry.updated",
            conversation_id=conversation_id,
            familiarity=new_fam,
            sarcasm_tolerance=new_sarcasm,
            playfulness=new_playfulness,
        )

        # Synthesize prompt notes
        notes = [
            f"Familiarity: {round(new_fam, 2)} ({'Stranger' if new_fam < 0.3 else 'Familiar Contact' if new_fam < 0.7 else 'Close Rapport'}).",
            f"Sarcasm Tolerance: {round(new_sarcasm, 2)} ({'Low - keep teasing mild' if new_sarcasm < 0.4 else 'High - enjoys sharp witty banter'}).",
        ]
        return notes
