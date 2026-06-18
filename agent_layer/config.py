# coding: utf-8

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

AGENT_LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent


class Settings(BaseSettings):
    """Agent runtime settings without business object construction."""

    llm_api_key: str
    llm_api_url: str
    llm_model: str = "qwen3-32b"
    llm_max_tokens: int = 1600  # Set the max output tokens per LLM response.
    llm_timeout_seconds: float = 60.0
    structured_output_retries: int = 1

    knowledge_base_url: str
    knowledge_base_timeout_seconds: float = 30.0

    stream_chunk_size: int = 32  # Max characters per streamed Agent transport chunk.
    recent_history_messages: int = 8  # Number of recent chat messages used for context resolution.
    context_message_chars: int = 500  # Max characters kept from each history/context message.

    retrieval_batch_size: int = 8  # Candidate evidence requested per retrieval call.
    max_retrieval_rounds: int = 3  # Max Self-RAG retrieval-assessment loops per question.
    max_follow_up_queries: int = 2  # Max model-proposed follow-up queries used per retrieval round.
    evidence_chunk_chars: int = 900  # Max characters kept from each evidence chunk in LLM prompts.
    evidence_context_chars: int = 8000  # Max total evidence characters sent to an LLM prompt.
    policy_internet_enabled: bool = True

    sql_statement_timeout_seconds: float = 10.0
    sql_max_rows: int = 200  # Max rows returned by a read-only SQL query.

    model_config = SettingsConfigDict(env_file=(PROJECT_ROOT / ".env").as_posix())

    @property
    def checkpoint_path(self) -> Path:  # LangGraph SQLite checkpoints for Agent workflows.
        return PROJECT_ROOT / "agent_layer/data/checkpoints/workflows.sqlite3"


settings = Settings()
