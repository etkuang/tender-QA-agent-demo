# coding: utf-8

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

AGENT_LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent


class Settings(BaseSettings):
    """Agent runtime settings without business object construction."""

    llm_api_key: str = "sk-u4ZJMre7fxQ7EdAqv7oRaC0Jc0wAfl980Er6kQrHKd304dxY"
    llm_api_url: str = "https://api.n1n.ai/v1"
    llm_model: str = "qwen3-32b"
    llm_max_tokens: int = 1600  # Set the max output tokens per LLM response.
    llm_timeout_seconds: float = 60.0
    structured_output_method: str = "json_mode"
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
    policy_preferred_domains: list[str] = ["gov.cn"]

    sql_statement_timeout_seconds: float = 10.0
    sql_max_rows: int = 200
    sql_max_joins: int = 4
    sql_max_subqueries: int = 4
    sql_dialect: str = "postgres"

    checkpoint_db_path: str = "agent_layer/data/checkpoints/workflows.sqlite3"

    no_results_response: str = "抱歉，当前可用数据源未检索到足以回答该问题的信息。"
    source_unavailable_response: str = "当前问题所需的数据源尚未配置或暂时不可用。"

    greeting_responses: dict[str, str] = {
        "你好": "您好！我是招投标六类智能问答助手，请问有什么可以帮您？",
        "您好": "您好！请问您想查询政策、招标、舆情、企业、价格还是商品信息？",
        "hi": "Hello！请问有什么可以帮您？",
        "hello": "Hello！请问有什么可以帮您？",
        "在吗": "在的，请问有什么可以帮您？",
        "谢谢": "不客气，有问题随时问我。",
        "感谢": "不客气。",
        "再见": "再见！如有问题，随时回来咨询。",
    }
    greeting_keywords: list[str] = ["你好", "您好", "hi", "hello", "嗨", "在吗", "在不在", "有人吗"]
    thanks_keywords: list[str] = ["谢谢", "感谢", "thanks", "thank"]
    goodbye_keywords: list[str] = ["再见", "拜拜", "bye", "goodbye"]

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env").as_posix(),
        case_sensitive=False,
        extra="ignore",
    )

    @staticmethod
    def _resolve_from_project(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def checkpoint_path(self) -> Path:
        return self._resolve_from_project(self.checkpoint_db_path)

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")


settings = Settings()
