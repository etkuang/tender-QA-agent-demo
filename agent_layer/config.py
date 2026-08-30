# coding: utf-8
# @Author: Wang Qingkang

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

AGENT_LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent


class Settings(BaseSettings):
    llm_provider: Literal["openai_compatible", "local_huggingface"] = "openai_compatible"
    llm_api_key: str | None = None
    llm_api_url: str | None = None
    llm_model: str = "qwen3-32b"
    llm_max_tokens: int = 1600
    llm_timeout_seconds: float = 60.0
    llm_concurrency_limit: int | None = None
    structured_output_retries: int = 1
    llm_local_device_map: str = "auto"
    llm_local_torch_dtype: str = "auto"
    llm_local_trust_remote_code: bool = False

    knowledge_base_url: str
    knowledge_base_timeout_seconds: float = 30.0

    recent_history_messages: int = 8
    context_message_chars: int = 500
    general_information_search_limit: int = 8
    evidence_chunk_chars: int = 900
    evidence_context_chars: int = 8000

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    @property
    def checkpoint_path(self) -> Path:
        return AGENT_LAYER_ROOT / "data/checkpoints/workflows.sqlite3"


settings = Settings()