"""Humor Engine: appropriateness gating, taxonomy selection, and directive synthesis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.humor.retrieval import HumorRetriever
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import RelationshipRecord, RelationshipRepository

logger = get_logger("humor.engine")

DISTRESS_PATTERNS = [
    re.compile(r"\b(funeral|died|death|passed\s+away|hospital|surgery)\b", re.IGNORECASE),
    re.compile(r"\b(depressed|crying|heartbroken|devastated|lost\s+my\s+job|laid\s+off)\b", re.IGNORECASE),
    re.compile(r"\b(having\s+a\s+panic\s+attack|can't\s+take\s+this\s+anymore)\b", re.IGNORECASE),
]


@dataclass
class HumorDecision:
    is_humor_appropriate: bool
    category: str
    intensity: float
    callback_to_reference: str | None
    directive_text: str


class HumorEngine:
    """Evaluates contextual humor appropriateness, modulates intensity, and synthesizes prompt directives."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()
        self.rel_repo = RelationshipRepository(self.db)
        self.retriever = HumorRetriever(self.db)
        self.retriever.seed_exemplars_if_empty()

    def evaluate_humor(
        self,
        conversation_id: str,
        user_message: str,
    ) -> HumorDecision:
        """Run complete humor decision pipeline."""
        # 1. Appropriateness & Distress Check
        is_distressed = any(p.search(user_message) for p in DISTRESS_PATTERNS)
        if is_distressed:
            logger.info("humor.suppressed_due_to_distress", conversation_id=conversation_id)
            return HumorDecision(
                is_humor_appropriate=False,
                category="none",
                intensity=0.0,
                callback_to_reference=None,
                directive_text="SUPPRESS ALL HUMOR. The user is expressing vulnerability, distress, or grief. Respond with calm, understated presence and authentic warmth. Do not use generic corporate platitudes.",
            )

        # 2. Retrieve relationship chemistry
        rel = self.rel_repo.get_or_create(conversation_id)

        # 3. Calculate max allowable intensity: I_max = 0.2 + (0.5 * T_sarcasm) + (0.3 * F)
        max_intensity = round(0.2 + (0.5 * rel.sarcasm_tolerance) + (0.3 * rel.familiarity), 2)
        max_intensity = min(1.0, max(0.2, max_intensity))

        # 4. Check for active running joke callbacks
        callback_id: str | None = None
        callback_hint: str | None = None
        try:
            jokes = json.loads(rel.running_jokes_json)
            for j in jokes:
                keywords = j.get("trigger_keywords", [])
                if any(k.lower() in user_message.lower() for k in keywords):
                    callback_id = j.get("callback_id")
                    callback_hint = j.get("punchline_hint")
                    break
        except Exception:
            pass

        # 5. Determine humor category
        if callback_id and callback_hint:
            category = "callback_humor"
        elif rel.sarcasm_tolerance >= 0.6 and rel.familiarity >= 0.4:
            category = "sarcastic_teasing"
        elif "why" in user_message.lower() or "how" in user_message.lower():
            category = "absurd_surreal"
        else:
            category = "dry_understated"

        # 6. Retrieve relevant exemplar
        exemplar = self.retriever.get_exemplar(category, max_intensity=max_intensity)

        # 7. Synthesize directive
        directive_parts = [
            f"Style: {category.replace('_', ' ').title()} (Intensity: {max_intensity}).",
        ]
        if callback_hint:
            directive_parts.append(f"Callback Reference: Subtly call back to: {callback_hint}.")
        if exemplar:
            directive_parts.append(
                f"Tone Inspiration (DO NOT copy words, use as cadence guide only): '{exemplar.exemplar_response}'"
            )

        directive_text = "\n".join(directive_parts)
        logger.info(
            "humor.directive_generated",
            category=category,
            intensity=max_intensity,
            callback=callback_id,
        )

        return HumorDecision(
            is_humor_appropriate=True,
            category=category,
            intensity=max_intensity,
            callback_to_reference=callback_id,
            directive_text=directive_text,
        )
