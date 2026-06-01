# coding: utf-8
# @Author: Wang Qingkang

from pathlib import Path

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deterministic cross-platform path resolution
# Prevents context drifting when launched from different working directories
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent


class AppSettings(BaseSettings):
    """
    Centralized Application Settings Management.
    Validates types and environments upon instantiation (Fail-Fast principle).
    """
    # ---------------- Global Configuration ----------------
    ENV: str = "development"
    DEBUG_MODE: bool = False

    # ---------------- Backend Infrastructure ----------------
    AGENT_BASE_URL: HttpUrl
    AGENT_API_KEY: str = "tender-agent"
    AGENT_MODEL_NAME: str = "tender-agent"

    # ---------------- Frontend Service Discovery ----------------
    BACKEND_URL: HttpUrl

    # ---------------- Network & Timeout Configuration ----------------
    @property
    def INTERNAL_TIMEOUT(self) -> float | None:
        """ Timeout for internal microservice RPC (Frontend -> Backend)"""
        return None if self.DEBUG_MODE else 1.0

    @property
    def EXTERNAL_TIMEOUT(self) -> float | None:
        """ Timeout for external LLM Provider API requests"""
        return None if self.DEBUG_MODE else 3.0

    @property
    def STREAM_TIMEOUT(self) -> float | None:
        """ Extended timeout for SSE streaming connections"""
        return None if self.DEBUG_MODE else 300.0

    # ---------------- Deterministic Paths (Cross-Platform) ----------------
    @property
    def db_dir(self) -> Path:
        """ Absolute path to the database directory."""
        return PROJECT_ROOT / "history"

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
