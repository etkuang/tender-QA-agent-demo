# coding: utf-8
# @Author: Wang Qingkang

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


KNOWLEDGE_BASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = KNOWLEDGE_BASE_ROOT.parent
MODEL_CACHE_ROOT = KNOWLEDGE_BASE_ROOT / "data" / "model_cache"

os.environ.setdefault("HF_HOME", (MODEL_CACHE_ROOT / "hf_home").as_posix())
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", (MODEL_CACHE_ROOT / "hub").as_posix())
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", (MODEL_CACHE_ROOT / "sentence_transformers").as_posix())


class Settings(BaseSettings):
    app_title: str = "Tender QA Knowledge Base"
    app_version: str = "1.0"

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
        "source_kind",
        "source_as_of",
        "data_version",
        "metadata",
    ]
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_bm25_k1: float = 1.2
    milvus_bm25_b: float = 0.75

    policy_collection_alias: str = "tender_qa_policy"
    top_k: int = 5
    vector_recall: int = 30
    bm25_recall: int = 30
    fusion_strategy: str = "rrf"
    fusion_dense_weight: float = 0.5
    fusion_keyword_weight: float = 0.5
    fusion_rrf_k: int = 60
    parent_context_enabled: bool = True
    reranker_enabled: bool = False
    reranker_candidate_pool: int = 20

    pdf_manifest_path: str = "knowledge_base_layer/data/pdf_sources.json"
    pdf_chunk_size: int = 1400
    pdf_chunk_overlap: int = 200
    policy_child_chunk_size: int = 700
    policy_child_chunk_overlap: int = 100

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env").as_posix(),
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def embedding_model_name_or_path(self) -> str:
        snapshots = MODEL_CACHE_ROOT / "sentence_transformers" / f"models--{self.embedding_model.replace('/', '--')}" / "snapshots"
        if snapshots.exists():
            candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if candidates:
                return candidates[-1].as_posix()
        return self.embedding_model

    @property
    def pdf_manifest(self) -> Path:
        path = Path(self.pdf_manifest_path).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


settings = Settings()