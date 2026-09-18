"""Character persona definition and YAML configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ConfigurationException
from app.core.logging import get_logger

logger = get_logger("ai.persona")


class IdentityConfig(BaseModel):
    type: str = "fictional_ai_character"
    name: str
    handle: str
    fictional_premise: str
    impersonation_disclaimer: str


class PersonalityConfig(BaseModel):
    confidence: str = "high"
    playfulness: str = "high"
    sarcasm: str = "high"
    warmth: str = "medium"
    curiosity: str = "high"
    neuroticism: str = "low"


class HumorConfig(BaseModel):
    dark: str = "high"
    dry: str = "very_high"
    absurd: str = "high"
    sarcastic: str = "high"
    wholesome: str = "low"
    generic_ai_humor: str = "forbidden"
    forced_jokes: str = "forbidden"


class CommunicationConfig(BaseModel):
    formality: str = "very_low"
    conversational: str = "very_high"
    verbosity: str = "low"
    emoji_usage: str = "occasional"
    casing: str = "lowercase"
    ending_periods: str = "forbidden"
    allowed_emojis: list[str] = Field(default_factory=lambda: ["😭", "💀", "😂"])
    forbidden_emojis: list[str] = Field(
        default_factory=lambda: ["🤗", "🤤", "😉", "😊", "🥰", "😜", "😝", "🤪", "😇", "👍"]
    )


class BehaviorConfig(BaseModel):
    prefer: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class AvailabilitySettings(BaseModel):
    departure_warning_minutes: int = 15


class SleepSettings(BaseModel):
    start: str = "23:00"
    end: str = "07:00"


class CharacterProfile(BaseModel):
    identity: IdentityConfig
    personality: PersonalityConfig
    humor: HumorConfig
    communication: CommunicationConfig
    behavior: BehaviorConfig
    availability: AvailabilitySettings = Field(default_factory=AvailabilitySettings)
    sleep: SleepSettings = Field(default_factory=SleepSettings)
    academics: dict[str, Any] = Field(default_factory=dict)
    routine: dict[str, Any] = Field(default_factory=dict)


@lru_cache
def load_character_profile(yaml_path: Path | str | None = None) -> CharacterProfile:
    """Load and validate the character profile from YAML."""
    if yaml_path is None:
        settings = get_settings()
        target_path = settings.resolved_character_config_path
    else:
        target_path = Path(yaml_path)

    if not target_path.exists():
        raise ConfigurationException(f"Character configuration file not found at: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not raw_data or "character" not in raw_data:
            raise ConfigurationException("Missing root 'character' key in YAML.")

        profile = CharacterProfile(**raw_data["character"])
        logger.info("persona.loaded_successfully", character_name=profile.identity.name)
        return profile
    except Exception as e:
        logger.error("persona.load_failed", path=str(target_path), error=str(e))
        raise ConfigurationException(f"Failed to parse character profile: {e}") from e
