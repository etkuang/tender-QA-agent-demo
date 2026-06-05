# coding: utf-8
# @Author: Wang Qingkang

import json
import os
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = FRONTEND_ROOT / "config.json"


class FrontendSettings:
    """Frontend runtime settings loaded from environment variables."""

    BACKEND_URL = os.environ["BACKEND_URL"]

    @property
    def INTERNAL_TIMEOUT(self):
        """Timeout for Frontend -> Backend metadata requests."""
        return None if os.environ.get("DEBUG_MODE") == "true" else 1.0

    @property
    def STREAM_TIMEOUT(self):
        """Timeout for Frontend -> Backend streaming requests."""
        return None if os.environ.get("DEBUG_MODE") == "true" else 300.0


settings = FrontendSettings()


def load_frontend_config() -> dict:
    """Load frontend configuration from app_frontend_layer/config.json."""
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_frontend_config(config_data: dict) -> None:
    """Persist frontend configuration to app_frontend_layer/config.json."""
    CONFIG_PATH.write_text(
        json.dumps(config_data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )