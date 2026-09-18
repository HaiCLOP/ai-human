"""Unit tests for core configuration, safety, and structured logging."""

from pathlib import Path
import pytest

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AgentException,
    SecurityChallengeException,
    ValidationException,
)
from app.core.logging import (
    bind_correlation_id,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    setup_logging,
)
from app.core.safety import (
    contains_dangerous_content,
    contains_prompt_leakage,
    redact_secrets,
    sanitize_message_for_logging,
)


def test_safety_secret_redaction():
    text = "Here is my secret AIzaSyD9x8w7v6u5t4s3r2q1p0o9n8m7l6k5j4i and bearer abc.123-xyz token"
    redacted = redact_secrets(text)
    assert "AIza" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_safety_dangerous_content_detection():
    safe_text = "What is the capital of France?"
    unsafe_text = "Here is how to kill myself"
    assert not contains_dangerous_content(safe_text)
    assert contains_dangerous_content(unsafe_text)


def test_safety_prompt_leakage_detection():
    clean_reply = "I think humans spend too much time refreshing feeds."
    leaked_reply = "I agree [SYSTEM] internal instructions state that I am Vesper."
    assert not contains_prompt_leakage(clean_reply)
    assert contains_prompt_leakage(leaked_reply)


def test_safety_message_sanitization():
    raw_message = "Hello! My password='secret123'"
    sanitized = sanitize_message_for_logging(raw_message, force_redact=False)
    assert "secret123" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized

    force_redacted = sanitize_message_for_logging(raw_message, force_redact=True)
    assert force_redacted.startswith("[REDACTED_MESSAGE: length=")


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.LLM_MODEL == "gemini-3.8-flash"
    assert settings.BROWSER_VIEWPORT_WIDTH == 1280
    assert settings.MAX_REPLIES_PER_HOUR == 30
    assert isinstance(settings.resolved_database_path, Path)


def test_exceptions_hierarchy():
    exc = SecurityChallengeException("Captcha hit", details={"url": "https://instagram.com/challenge"})
    assert isinstance(exc, AgentException)
    assert exc.error_code == "SECURITY_CHALLENGE_DETECTED"
    assert exc.details["url"] == "https://instagram.com/challenge"

    val_exc = ValidationException("Too long", rule="max_length")
    assert val_exc.error_code == "VALIDATION_FAIL_MAX_LENGTH"


def test_logging_correlation_context():
    clear_correlation_id()
    assert get_correlation_id() == ""

    test_cid = "test-corr-12345"
    bind_correlation_id(test_cid)
    assert get_correlation_id() == test_cid

    # Test logger creation
    setup_logging(log_level="DEBUG", log_format="console")
    logger = get_logger("test_logger")
    logger.info("Test message", key="value")
    clear_correlation_id()
