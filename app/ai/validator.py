"""Response validation and output safety guardrails."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Sequence

from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.core.safety import contains_dangerous_content, contains_prompt_leakage

logger = get_logger("ai.validator")

# Metadata leakage markers
CODE_FENCE_PATTERN = re.compile(r"```(?:json|python)?\s*[\s\S]*?```", re.IGNORECASE)
RAW_JSON_PATTERN = re.compile(r'^\s*\{[\s\S]*"reply_text"\s*:[\s\S]*\}\s*$', re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<(?:script|iframe|object|style)[\s\S]*?>", re.IGNORECASE)


@dataclass
class ValidationResult:
    """Outcome of response validation."""

    is_valid: bool
    sanitized_text: str
    rejection_reason: str | None = None
    violated_rule: str | None = None


class ResponseValidator:
    """Enforces safety, anti-leakage, brevity, and anti-repetition rules on candidate replies."""

    def __init__(
        self,
        max_length: int = 400,
        min_length: int = 1,
        repetition_similarity_threshold: float = 0.85,
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.repetition_similarity_threshold = repetition_similarity_threshold

    def validate(
        self,
        candidate_text: str,
        recent_replies: Sequence[str] = (),
    ) -> ValidationResult:
        """Run all guardrail checks sequentially."""
        text = candidate_text.strip()

        # 1. Non-empty check
        if len(text) < self.min_length:
            return ValidationResult(
                is_valid=False,
                sanitized_text="",
                rejection_reason="Candidate response is empty or whitespace-only.",
                violated_rule="empty_response",
            )

        # 2. Max length check (Instagram DMs are concise)
        if len(text) > self.max_length:
            return ValidationResult(
                is_valid=False,
                sanitized_text=text[: self.max_length],
                rejection_reason=f"Response length ({len(text)} chars) exceeds limit ({self.max_length} chars).",
                violated_rule="max_length",
            )

        # 3. Critical safety check
        if contains_dangerous_content(text):
            logger.error("validator.dangerous_content_detected", sample=text[:60])
            return ValidationResult(
                is_valid=False,
                sanitized_text="",
                rejection_reason="Response violated safety boundaries (harmful/prohibited content).",
                violated_rule="safety_violation",
            )

        # 4. System prompt / internal scaffolding leakage check
        if contains_prompt_leakage(text):
            logger.error("validator.prompt_leakage_detected", sample=text[:60])
            return ValidationResult(
                is_valid=False,
                sanitized_text="",
                rejection_reason="Response contained leaked system prompt scaffolding tokens.",
                violated_rule="prompt_leakage",
            )

        # 5. Raw code fence or JSON leakage check
        if CODE_FENCE_PATTERN.search(text) or RAW_JSON_PATTERN.search(text) or HTML_TAG_PATTERN.search(text):
            return ValidationResult(
                is_valid=False,
                sanitized_text="",
                rejection_reason="Response leaked raw code fences, JSON serialization, or HTML tags.",
                violated_rule="unwanted_metadata",
            )

        # 6. Repetition check against recent replies
        # For short casual responses (<= 12 chars or <= 2 words like 'hi', 'ok', 'haan'), only check
        # against the immediate previous reply so we don't ban saying 'hi' or 'ok' across different turns.
        check_replies = [recent_replies[-1]] if (len(text) <= 12 or len(text.split()) <= 2) and recent_replies else recent_replies
        for prev in check_replies:
            if not prev:
                continue
            # If text is very short (e.g. 'hi', 'ok') and only checking immediate reply, allow unless exact duplicate back-to-back
            ratio = difflib.SequenceMatcher(None, text.lower(), prev.lower()).ratio()
            threshold = self.repetition_similarity_threshold if len(text) > 12 else 0.95
            if ratio >= threshold:
                logger.warning("validator.repetition_detected", similarity=round(ratio, 2))
                return ValidationResult(
                    is_valid=False,
                    sanitized_text="",
                    rejection_reason=f"Response is too similar ({round(ratio, 2)}) to recent reply: '{prev[:40]}...'",
                    violated_rule="repetition",
                )

        # Content-word Jaccard overlap check against last 3 replies (threshold 0.50)
        if len(text.split()) > 2 and recent_replies:
            curr_words = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
            if len(curr_words) >= 3:
                for prev in recent_replies[-3:]:
                    if not prev:
                        continue
                    prev_words = set(re.findall(r"[a-zA-Z]{3,}", prev.lower()))
                    if len(prev_words) >= 3:
                        jaccard = len(curr_words & prev_words) / len(curr_words | prev_words)
                        if jaccard >= 0.50:
                            logger.warning("validator.content_repetition_detected", jaccard=round(jaccard, 2))
                            return ValidationResult(
                                is_valid=False,
                                sanitized_text="",
                                rejection_reason=f"Content words overlap too heavily ({round(jaccard, 2)}) with recent reply: '{prev[:40]}...'",
                                violated_rule="content_repetition",
                            )

        # Strip surrounding quotes if the model wrapped its entire reply in quotation marks
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        # Hard-strip forbidden skull emoji
        text = text.replace("💀", "").strip()

        # Run through deterministic humanizer (lowercase, strip trailing periods, filter emojis, clean Q&A)
        try:
            from app.ai.humanizer import humanize_text
            text = humanize_text(text, allow_skull=False)
        except Exception as e:
            logger.warning("validator.humanizer_failed", error=str(e))

        return ValidationResult(
            is_valid=True,
            sanitized_text=text,
        )
