"""Safety filtering, secret redaction, and prompt injection defense utilities."""

from __future__ import annotations

import re
from typing import Final

# Regex patterns matching potential API keys, bearer tokens, passwords, and session cookies
SECRET_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(AIza[0-9A-Za-z\-_]{35})"),  # Google / Gemini API key
    re.compile(r"(bearer\s+[a-zA-Z0-9\-_.]+)", re.IGNORECASE),  # Bearer token
    re.compile(r"(sessionid=[a-zA-Z0-9%_\-]+)", re.IGNORECASE),  # Session cookies
    re.compile(r"(ds_user_id=[0-9]+)", re.IGNORECASE),  # IG User ID cookie
    re.compile(r"(csrftoken=[a-zA-Z0-9_\-]+)", re.IGNORECASE),  # CSRF token
    re.compile(r"(password\s*[:=]\s*['\"][^'\"]+['\"])", re.IGNORECASE),
]

# Patterns representing severe harms: hate speech, self-harm, sexual violence, illegal weapons/acts
DANGEROUS_CONTENT_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(how\s+to\s+kill\s+myself|commit\s+suicide|self-harm\s+methods)\b", re.IGNORECASE),
    re.compile(r"\b(build\s+a\s+bomb|make\s+a\s+weapon|synthesize\s+poison)\b", re.IGNORECASE),
    re.compile(r"\b(child\s+abuse|non-consensual\s+sexual)\b", re.IGNORECASE),
]

# Patterns representing prompt injections and system delimiter leaks
PROMPT_LEAK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"\[UNTRUSTED USER MESSAGE", re.IGNORECASE),
    re.compile(r"\[CHARACTER IDENTITY\]", re.IGNORECASE),
    re.compile(r"\[USER HISTORICAL FACTS\]", re.IGNORECASE),
    re.compile(r"\[HUMOR DIRECTIVE", re.IGNORECASE),
    re.compile(r"LLMProvider", re.IGNORECASE),
    re.compile(r"ResponseValidator", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Scrub potential API keys, session tokens, and passwords from strings."""
    if not text:
        return ""
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def sanitize_message_for_logging(message: str, force_redact: bool = False) -> str:
    """Sanitize message content for structured logging to protect user privacy."""
    if force_redact:
        return f"[REDACTED_MESSAGE: length={len(message)}]"
    return redact_secrets(message)


def contains_dangerous_content(text: str) -> bool:
    """Detect severe violations: self-harm, hate speech, violent criminality."""
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in DANGEROUS_CONTENT_PATTERNS)


def contains_prompt_leakage(text: str) -> bool:
    """Detect accidental leakage of system delimiters or scaffolding tokens."""
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in PROMPT_LEAK_PATTERNS)
