"""Structured logging setup using structlog with secret scrubbing and correlation tracking."""

from __future__ import annotations

import contextvars
import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.safety import redact_secrets

# ContextVar for tracing a single message turn across all subsystems
correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation ID to the current async context."""
    correlation_id_ctx.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID from context."""
    correlation_id_ctx.set("")


def get_correlation_id() -> str:
    """Get active correlation ID or return empty string."""
    return correlation_id_ctx.get()


def _add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Processor to inject correlation ID if present."""
    cid = get_correlation_id()
    if cid and "correlation_id" not in event_dict:
        event_dict["correlation_id"] = cid
    return event_dict


def _secret_scrubber_processor(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Processor that redacts secrets from any string values in event_dict."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact_secrets(value)
        elif isinstance(value, dict):
            # Recursively redact dict fields
            event_dict[key] = {
                k: redact_secrets(v) if isinstance(v, str) else v
                for k, v in value.items()
            }
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file_path: Path | None = None,
) -> None:
    """Configure standard library logging and structlog."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _secret_scrubber_processor,
    ]

    # Configure root standard library logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if log_format == "json":
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler (always JSONL)
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "agent") -> structlog.stdlib.BoundLogger:
    """Retrieve a bound structlog logger."""
    return structlog.get_logger(name)
