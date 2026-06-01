# coding: utf-8
# @Author: Wang Qingkang


import json
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = FRONTEND_ROOT / "config.json"


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