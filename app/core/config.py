"""Application settings, environment variables, and configuration management."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Instagram AI Character agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    # LLM Settings
    LLM_PROVIDER: str = Field(default="gemini", description="AI provider identifier: 'gemini', 'groq', 'openrouter', 'mock'")
    LLM_MODEL: str = Field(default="gemini-3.8-flash", description="Model name/identifier")
    LLM_FALLBACK_MODEL: str = Field(default="gemini-3.6-flash", description="Fallback model if primary experiences capacity issues")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    LLM_TEMPERATURE: float = Field(default=0.85, description="Sampling temperature for creative humor")
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=350, description="Max tokens for output (sufficient for short chat responses)")
    LLM_TIMEOUT_SECONDS: float = Field(default=20.0, description="Network timeout for LLM API calls")
    LLM_MAX_RETRIES: int = Field(default=3, description="Maximum retry attempts on 429/5xx errors")

    # Persistence & Storage
    DATABASE_PATH: Path = Field(default=Path("data/agent.db"), description="Path to SQLite database file")

    # Browser Automation
    BROWSER_PROFILE_PATH: Path = Field(default=Path("data/browser_profile"), description="Persistent Chromium context")
    BROWSER_HEADLESS: bool = Field(default=False, description="Run Chromium in headless mode")
    BROWSER_VIEWPORT_WIDTH: int = Field(default=1280)
    BROWSER_VIEWPORT_HEIGHT: int = Field(default=800)
    BROWSER_NAVIGATION_TIMEOUT_MS: int = Field(default=30000)

    # Instagram Targeting & Rate Limits
    TARGET_THREAD_ID: str = Field(default="", description="Optional specific DM thread ID to monitor")
    TARGET_USERNAME: str = Field(default="", description="Optional target user handle to monitor")
    MONITOR_ALL_INBOX: bool = Field(default=False, description="Whether to poll entire inbox for new messages")
    INBOX_POLL_INTERVAL_SECONDS: float = Field(default=5.0, description="Polling interval in seconds")
    MAX_REPLIES_PER_HOUR: int = Field(default=30, description="Safety throttle: maximum dispatched replies/hr")
    MIN_REPLY_DELAY_SECONDS: float = Field(default=3.0, description="Random delay floor before dispatching reply")
    MAX_REPLY_DELAY_SECONDS: float = Field(default=7.0, description="Random delay ceiling before dispatching reply")

    # Observability & Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    LOG_FORMAT: Literal["console", "json"] = Field(default="console")
    LOG_SANITIZE_MESSAGES: bool = Field(default=False, description="Mask message contents in structured logs")
    LOG_FILE_PATH: Path = Field(default=Path("logs/agent.jsonl"))

    # Local RAG & Embeddings
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", description="FastEmbed ONNX model")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.65, description="Minimum cosine similarity for inclusion")
    RAG_TOP_K: int = Field(default=2, description="Maximum retrieved chunks to inject into prompt")

    # Historical Conversation Intelligence
    HISTORICAL_DATASET_PATH: Path = Field(default=Path("data/instagram_history/conversations"), description="Path to Instagram JSON export conversations")
    HISTORICAL_BASE_PATH: Path = Field(default=Path("data/instagram_history"), description="Base path for processed historical data, profiles, and reports")
    OPERATOR_NAMES: list[str] = Field(default_factory=lambda: ["Arnav Srivastava", "Arnav"], description="Configured operator display names / aliases")
    HISTORICAL_LOCAL_ONLY: bool = Field(default=True, description="Strict local-only mode; raw chats never uploaded")
    HISTORICAL_RECENT_WINDOW_DAYS: int = Field(default=90, description="Window in days for recent style calculation")
    HISTORICAL_LONG_TERM_WEIGHT: float = Field(default=0.60, description="Weight for long-term historical style")
    HISTORICAL_RECENT_WEIGHT: float = Field(default=0.40, description="Weight for recent historical style")

    # Character Config
    CHARACTER_CONFIG_PATH: Path = Field(default=Path("config/character.yaml"))

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative path against BASE_DIR."""
        if path.is_absolute():
            return path
        return (self.BASE_DIR / path).resolve()

    @property
    def resolved_database_path(self) -> Path:
        return self.resolve_path(self.DATABASE_PATH)

    @property
    def resolved_browser_profile_path(self) -> Path:
        return self.resolve_path(self.BROWSER_PROFILE_PATH)

    @property
    def resolved_log_file_path(self) -> Path:
        return self.resolve_path(self.LOG_FILE_PATH)

    @property
    def resolved_character_config_path(self) -> Path:
        return self.resolve_path(self.CHARACTER_CONFIG_PATH)

    @property
    def resolved_historical_dataset_path(self) -> Path:
        return self.resolve_path(self.HISTORICAL_DATASET_PATH)

    @property
    def resolved_historical_base_path(self) -> Path:
        return self.resolve_path(self.HISTORICAL_BASE_PATH)


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached singleton application settings."""
    return Settings()
