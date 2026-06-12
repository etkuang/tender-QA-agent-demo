# coding: utf-8

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


AGENT_LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent
MODEL_CACHE_ROOT = AGENT_LAYER_ROOT / "data" / "model_cache"

os.environ.setdefault("HF_HOME", (MODEL_CACHE_ROOT / "hf_home").as_posix())
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", (MODEL_CACHE_ROOT / "hub").as_posix())
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", (MODEL_CACHE_ROOT / "sentence_transformers").as_posix())


class Settings(BaseSettings):
    """Agent runtime settings without business object construction."""

    app_title: str = "Tender QA Multi-Workflow Agent"
    app_version: str = "1.0"
    app_description: str = "Six-domain tender and bidding question answering agent."

    llm_api_key: str = ""
    llm_api_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1600
    llm_timeout_seconds: float = 60.0
    structured_output_method: str = "json_mode"
    structured_output_retries: int = 1

    embedding_model: str = "moka-ai/m3e-base"
    embedding_dimension: int = 768

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_database: str = "tender_qa"
    milvus_timeout_seconds: float = 10.0
    milvus_metric_type: str = "COSINE"
    milvus_dense_field: str = "dense_vector"
    milvus_sparse_field: str = "sparse_vector"
    milvus_primary_field: str = "id"
    milvus_content_field: str = "content"
    milvus_output_fields: list[str] = [
        "content",
        "title",
        "source_url",
        "document_id",
        "law_name",
        "article_id",
        "entity_id",
        "published_at",
        "authority_level",
        "freshness_level",
        "parent_id",
        "chunk_type",
        "validity_status",
        "region",
        "effective_date",
        "end_date",
        "data_version",
        "metadata",
    ]
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_bm25_k1: float = 1.2
    milvus_bm25_b: float = 0.75
    policy_collection: str = "tender_qa_policy"
    tender_collection: str = "tender_qa_tender"
    public_opinion_collection: str = "tender_qa_public_opinion"
    company_collection: str = "tender_qa_company"
    price_collection: str = "tender_qa_price"
    product_collection: str = "tender_qa_product"
    legacy_chroma_dir: str = "agent_layer/data/databases/chroma_db"

    routing_confidence_high: float = 0.80
    routing_confidence_low: float = 0.45
    stream_chunk_size: int = 32
    recent_history_messages: int = 8
    context_message_chars: int = 500

    top_k: int = 5
    vector_recall: int = 30
    bm25_recall: int = 30
    fusion_strategy: str = "rrf"
    fusion_dense_weight: float = 0.5
    fusion_keyword_weight: float = 0.5
    fusion_rrf_k: int = 60
    summarize_max_chunks: int = 8
    summarize_chunk_length: int = 900
    parent_context_enabled: bool = True
    reranker_enabled: bool = False
    reranker_candidate_pool: int = 20
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
    def legacy_chroma_path(self) -> Path:
        return self._resolve_from_project(self.legacy_chroma_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self._resolve_from_project(self.checkpoint_db_path)

    @property
    def embedding_model_name_or_path(self) -> str:
        snapshots = MODEL_CACHE_ROOT / "sentence_transformers" / f"models--{self.embedding_model.replace('/', '--')}" / "snapshots"
        if snapshots.exists():
            candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if candidates:
                return candidates[-1].as_posix()
        return self.embedding_model

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")


settings = Settings()
