"""Operator identification and pseudonymous contact identity management."""

from __future__ import annotations

import hashlib
from typing import Sequence

from app.core.config import get_settings
from app.core.exceptions import AgentException
from app.core.logging import get_logger

logger = get_logger("learning.operator_detector")


class AmbiguousOperatorException(AgentException):
    """Raised when the operator cannot be confidently identified among participants."""

    def __init__(self, message: str, participants: list[str], configured_aliases: list[str]):
        super().__init__(
            message=message,
            error_code="AMBIGUOUS_OPERATOR_IDENTITY",
            details={"participants": participants, "configured_aliases": configured_aliases},
        )


class OperatorDetector:
    """Identifies the human operator and maintains pseudonymous contact mappings."""

    def __init__(self, operator_names: Sequence[str] | None = None):
        settings = get_settings()
        self.operator_aliases = [
            name.strip().lower() for name in (operator_names or settings.OPERATOR_NAMES) if name.strip()
        ]

    def is_operator_name(self, name: str) -> bool:
        """Check if a participant name matches any operator alias."""
        clean = name.strip().lower()
        return any(clean == alias or clean.startswith(alias) for alias in self.operator_aliases)

    def identify_participants(
        self,
        participants: list[str],
        conversation_file: str = "",
    ) -> tuple[str, list[str]]:
        """Identify operator and contacts from participant list.

        Returns:
            tuple[operator_name, list_of_contact_names]

        Raises:
            AmbiguousOperatorException: If operator cannot be identified with 100% certainty.
        """
        matched_operators: list[str] = []
        contacts: list[str] = []

        for p in participants:
            if self.is_operator_name(p):
                matched_operators.append(p)
            else:
                contacts.append(p)

        if len(matched_operators) == 0:
            msg = (
                f"STOP IMPORT: Could not identify operator in participants {participants} "
                f"for conversation '{conversation_file}'. Please configure operator names in character.yaml."
            )
            logger.error("operator_detector.operator_not_found", participants=participants)
            raise AmbiguousOperatorException(msg, participants, self.operator_aliases)

        if len(matched_operators) > 1:
            # Check if they are just identical names
            unique_matched = set(matched_operators)
            if len(unique_matched) > 1:
                msg = (
                    f"STOP IMPORT: Ambiguous operator identity in conversation '{conversation_file}'. "
                    f"Multiple participants matched operator aliases: {matched_operators}."
                )
                logger.error("operator_detector.ambiguous_operator", matches=matched_operators)
                raise AmbiguousOperatorException(msg, participants, self.operator_aliases)

        operator_name = matched_operators[0]
        return operator_name, contacts

    @staticmethod
    def generate_contact_id(display_name: str) -> str:
        """Generate a deterministic pseudonymous internal contact ID (e.g. contact_7f2c9a)."""
        clean = display_name.strip().lower()
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:8]
        return f"contact_{digest}"
