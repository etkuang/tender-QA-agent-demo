# coding: utf-8
# @Author: Wang Qingkang

from pathlib import Path

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deterministic cross-platform path resolution
# Prevents context drifting when launched from different working directories
CORE_DIR = Path(__file__).resolve().parent
BACKEND_LAYER_ROOT = CORE_DIR.parent


class AppSettings(BaseSettings):
    """
    Centralized Application Settings Management.
    Validates types and environments upon instantiation (Fail-Fast principle).
    """
    ENV: str = "development"
    DEBUG_MODE: bool = False

    AGENT_BASE_URL: HttpUrl

    @property
    def STREAM_TIMEOUT(self) -> float | None:
        """Extended timeout for Backend -> Agent streaming connections."""
        return None if self.DEBUG_MODE else 300.0

    @property
    def db_dir(self) -> Path:
        """Absolute path to the database directory."""
        return BACKEND_LAYER_ROOT / "history"

    @property
    def db_path(self) -> Path:
        """ Absolute path to the SQLite/aiosqlite database file."""
        return self.db_dir / "history.db"

    # Pydantic Settings Metadata Configuration
    model_config = SettingsConfigDict(
        extra="ignore"  # Bypass unexpected environment variables safely
    )


# Instantiate a singleton instance for global access
settings = AppSettings()
