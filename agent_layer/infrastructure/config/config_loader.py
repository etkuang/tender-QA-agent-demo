from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    _instance = None
    _config = {}
    _loaded_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = "config.yaml", force_reload: bool = False) -> dict:
        if self._config and not force_reload:
            return self._config

        path = Path(config_path)

        if not path.exists():
            raise ValueError(f"Config file does not exist: {path}")

        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle) or {}

        if not isinstance(parsed, dict):
            raise ValueError(f"Config root must be a mapping: {path}")

        self._config = parsed
        self._loaded_path = path
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default

        keys = key.split(".")
        value = self._config
        try:
            for item in keys:
                value = value[item]
            return value
        except (KeyError, TypeError):
            return default

    def is_loaded(self) -> bool:
        return bool(self._config)

    def loaded_path(self) -> Path | None:
        return self._loaded_path

    def dump(self) -> dict:
        return dict(self._config)


config_loader = ConfigLoader()