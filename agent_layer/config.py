# coding: utf-8
# @Author: Wang Qingkang

from pathlib import Path
from typing import Literal

from pydantic import Field
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

    # Milvus server address; defaults to the local deployment.
    evidence_milvus_uri: str = "http://127.0.0.1:19530"
    # Empty means no authentication credentials are supplied.
    evidence_milvus_token: str = ""
    # Timeout for each Milvus SDK operation, in seconds.
    evidence_milvus_timeout_seconds: float = Field(default=30.0, gt=0)
    # Maximum concurrent encoding calls on the shared embedding model.
    embedding_concurrency_limit: int = Field(default=4, ge=1)
    # Unicode code-point limits; four UTF-8 bytes per character must fit in Milvus.
    evidence_title_max_chars: int = Field(default=120, ge=1, le=8192)
    evidence_content_max_chars: int = Field(default=2000, ge=1, le=8192)
    evidence_reliability_max_chars: int = Field(default=500, ge=1, le=8192)
    # Maximum recent history messages included in a model prompt.
    recent_history_messages: int = Field(default=8, ge=1)
    # Maximum characters retained from each history message in category prompts.
    history_message_chars: int = Field(default=500, ge=1)
    # Maximum characters retained from each prerequisite question in category prompts.
    prerequisite_question_chars: int = Field(default=500, ge=1)
    # Maximum characters retained from each prerequisite answer in category prompts.
    prerequisite_answer_chars: int = Field(default=500, ge=1)
    # Maximum total characters allocated to rendered prerequisite-answer blocks.
    prerequisite_context_chars: int = Field(default=2000, ge=1)
    # Maximum active Milvus evidence records returned for one question.
    evidence_pool_search_limit: int = Field(default=5, ge=1)
    # Maximum evidence records returned by one Internet search.
    internet_search_result_limit: int = Field(default=8, ge=1)
    # Maximum Internet-search tool rounds allowed for one question.
    internet_search_max_rounds: int = Field(default=2, ge=1)
    # Maximum evidence records rendered into a model prompt.
    evidence_context_limit: int = Field(default=10, ge=1)
    # Maximum content characters rendered for each evidence record.
    evidence_chunk_chars: int = Field(default=500, ge=1)
    # Maximum total characters allocated to rendered evidence blocks.
    evidence_context_chars: int = Field(default=2000, ge=1)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    @property
    def checkpoint_path(self) -> Path:
        return AGENT_LAYER_ROOT / "data/checkpoints/workflows.sqlite3"


settings = Settings()