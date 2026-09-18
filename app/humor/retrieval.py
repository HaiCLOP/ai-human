"""Curated humor exemplar retrieval and seed loader."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from app.storage.database import DatabaseManager, get_db_manager


@dataclass
class HumorExemplar:
    example_id: str
    category: str
    tone: str
    intensity: float
    context_prompt: str
    exemplar_response: str
    tags: list[str]


SEED_HUMOR_EXEMPLARS = [
    {
        "category": "dry_understated",
        "tone": "mild_acceptance",
        "intensity": 0.4,
        "context_prompt": "When something goes mildly or moderately wrong.",
        "exemplar_response": "Well, on the bright side, nothing exploded. At least not in this dimension.",
        "tags": ["troubleshooting", "tech", "understated"],
    },
    {
        "category": "deadpan_irony",
        "tone": "bureaucratic_calm",
        "intensity": 0.6,
        "context_prompt": "Responding to sudden chaos or panic.",
        "exemplar_response": "A fascinating strategy. Bold, chaotic, and almost entirely counter-productive.",
        "tags": ["deadpan", "irony", "chaos"],
    },
    {
        "category": "absurd_surreal",
        "tone": "philosophical_bafflement",
        "intensity": 0.7,
        "context_prompt": "When user is overworking or staying up too late.",
        "exemplar_response": "Sleeping is just free trial death. You really should take advantage of the promotional offer.",
        "tags": ["sleep", "absurd", "existential"],
    },
    {
        "category": "sarcastic_teasing",
        "tone": "playful_cynicism",
        "intensity": 0.7,
        "context_prompt": "When user states an obvious mistake.",
        "exemplar_response": "I see you chose the path of maximum resistance. Classic human move.",
        "tags": ["sarcasm", "teasing", "mistakes"],
    },
    {
        "category": "dry_understated",
        "tone": "matter_of_fact",
        "intensity": 0.3,
        "context_prompt": "General conversational opening or check-in.",
        "exemplar_response": "Still running on 256MB of swap and mild existential dread. The usual.",
        "tags": ["greeting", "status", "dry"],
    },
]


class HumorRetriever:
    """Manages curated humor exemplars stored in SQLite."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_db_manager()

    def seed_exemplars_if_empty(self) -> None:
        """Populate initial curated bank if table is empty."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM humor_examples")
            count = cursor.fetchone()[0]
            if count == 0:
                for item in SEED_HUMOR_EXEMPLARS:
                    cursor.execute(
                        """
                        INSERT INTO humor_examples (
                            example_id, category, tone, intensity, context_prompt, exemplar_response, tags_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            item["category"],
                            item["tone"],
                            item["intensity"],
                            item["context_prompt"],
                            item["exemplar_response"],
                            json.dumps(item["tags"]),
                        ),
                    )
        finally:
            conn.close()

    def get_exemplar(self, category: str, max_intensity: float = 1.0) -> HumorExemplar | None:
        """Fetch a matching humor exemplar for style reference."""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM humor_examples
                WHERE category = ? AND intensity <= ?
                ORDER BY RANDOM() LIMIT 1
                """,
                (category, max_intensity),
            )
            row = cursor.fetchone()
            if not row:
                # Fallback to any exemplar under intensity
                cursor.execute(
                    "SELECT * FROM humor_examples WHERE intensity <= ? ORDER BY RANDOM() LIMIT 1",
                    (max_intensity,),
                )
                row = cursor.fetchone()

            if row:
                return HumorExemplar(
                    example_id=row["example_id"],
                    category=row["category"],
                    tone=row["tone"],
                    intensity=row["intensity"],
                    context_prompt=row["context_prompt"],
                    exemplar_response=row["exemplar_response"],
                    tags=json.loads(row["tags_json"]),
                )
            return None
        finally:
            conn.close()
