import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_layer.infrastructure.config.config_loader import config_loader


CONFIG_DIR = Path(__file__).resolve().parent
AGENT_LAYER_ROOT = CONFIG_DIR.parent.parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent
CONFIG_FILE = CONFIG_DIR / "config.yaml"
config_loader.load(CONFIG_FILE.as_posix())

MODEL_CACHE_ROOT = PROJECT_ROOT / "agent_layer" / "data" / "model_cache"
os.environ.setdefault("HF_HOME", (MODEL_CACHE_ROOT / "hf_home").as_posix())
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", (MODEL_CACHE_ROOT / "hub").as_posix())
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", (MODEL_CACHE_ROOT / "sentence_transformers").as_posix())


class Settings(BaseSettings):
    """Runtime settings for the agent layer."""

    llm_api_key: str = "sk-BBl08Ecy36LiYoNHPv8GTUQ0ykOuKY0cK27kw7DURgY3HMXM"
    llm_api_url: str = "https://api.n1n.ai/v1"
    llm_model: str = "gpt-3.5-turbo"
    llm_max_tokens: int = 1000

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    app_title: str = "Tender QA RAG Agent"
    app_version: str = "2.1.0"
    app_description: str = "Tender and bidding regulation assistant."
    unrelated_response: str = "抱歉，我只能回答招投标相关问题。"
    no_results_response: str = "抱歉，未找到相关信息。"

    embedding_model: str = "moka-ai/m3e-base"
    top_k: int = 5
    vector_recall: int = 20
    fusion_strategy: str = "smart"

    boost_law_article: float = 0.08
    boost_article_start: float = 0.05
    boost_high_keyword: float = 0.06
    boost_medium_keyword: float = 0.04
    boost_low_keyword: float = 0.03
    boost_regulation_keyword: float = 0.05
    boost_multi_category: float = 0.02
    boost_max: float = 0.2

    penalty_min: float = -0.3
    semantic_law_article_penalty: float = 0.5
    min_final_score: float = 0.1
    high_norm_threshold: float = 0.8
    dense_no_boost_penalty: float = 0.12

    react_max_steps: int = 5
    react_temperature: float = 0.3

    summarize_max_chunks: int = 8
    summarize_chunk_length: int = 600
    low_score_threshold: float = 0.25

    enable_question_rewrite: bool = True

    agent_search_top_k: int = 3
    agent_search_max_text_len: int = 500
    agent_fallback_top_k: int = 3
    agent_article_max_results: int = 2
    agent_article_max_text_len: int = 800

    model_cache_dir: str = "agent_layer/data/model_cache"
    pdf_dir: str = "agent_layer/data/pdf"
    chroma_persist_dir: str = "agent_layer/data/databases/chroma_db"
    sqlite_db_path: str = "agent_layer/data/sessions.db"
    manual_eval_path: str = "agent_layer/data/manual_eval_set.json"
    eval_set_25_path: str = "agent_layer/data/eval_set_25.json"

    session_ttl_seconds: int = 604800
    max_messages_per_session: int = 30
    cleanup_interval_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env").as_posix(),
        case_sensitive=False,
        extra="allow",
    )

    @staticmethod
    def _resolve_from_project(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def model_cache_path(self) -> Path:
        return self._resolve_from_project(self.model_cache_dir)

    @property
    def pdf_dir_path(self) -> Path:
        return self._resolve_from_project(self.pdf_dir)

    @property
    def chroma_persist_dir_path(self) -> Path:
        return self._resolve_from_project(self.chroma_persist_dir)

    @property
    def sqlite_db_file(self) -> Path:
        return self._resolve_from_project(self.sqlite_db_path)

    @property
    def manual_eval_file(self) -> Path:
        return self._resolve_from_project(self.manual_eval_path)

    @property
    def eval_set_25_file(self) -> Path:
        return self._resolve_from_project(self.eval_set_25_path)

    def _yaml_get(self, key: str, default: Any = None) -> Any:
        return config_loader.get(key, default)

    @property
    def weights_keyword_heavy(self) -> tuple:
        value = self._yaml_get("retrieval.weights.keyword_heavy", [0.75, 0.25])
        return (value[0], value[1])

    @property
    def weights_semantic_heavy(self) -> tuple:
        value = self._yaml_get("retrieval.weights.semantic_heavy", [0.4, 0.6])
        return (value[0], value[1])

    @property
    def weights_balanced(self) -> tuple:
        value = self._yaml_get("retrieval.weights.balanced", [0.65, 0.35])
        return (value[0], value[1])

    @property
    def tender_keywords(self) -> dict:
        return self._yaml_get("retrieval.boost_keywords", {"high": [], "medium": [], "low": []})

    @property
    def regulation_keywords(self) -> dict:
        return self._yaml_get("retrieval.regulation_keywords", {"high": [], "medium": []})

    @property
    def penalty_patterns(self) -> list:
        return self._yaml_get("retrieval.penalty_patterns", [])

    @property
    def greeting_responses(self) -> dict:
        return self._yaml_get("intent.greeting_responses", {})

    @property
    def greeting_keywords(self) -> list:
        return self._yaml_get("intent.greeting_keywords", [])

    @property
    def thanks_keywords(self) -> list:
        return self._yaml_get("intent.thanks_keywords", [])

    @property
    def goodbye_keywords(self) -> list:
        return self._yaml_get("intent.goodbye_keywords", [])

    @property
    def unrelated_keywords(self) -> list:
        return self._yaml_get("intent.unrelated_keywords", [])

    @property
    def bidding_keywords(self) -> list:
        return self._yaml_get("intent.bidding_keywords", [])

    @property
    def keyword_heavy_patterns(self) -> list:
        return self._yaml_get("retrieval.keyword_heavy_patterns", [])

    @property
    def semantic_heavy_patterns(self) -> list:
        return self._yaml_get("retrieval.semantic_heavy_patterns", [])

    @property
    def default_chunk_size(self) -> int:
        return self._yaml_get("pdf.default_chunk_size", 800)

    @property
    def default_overlap(self) -> int:
        return self._yaml_get("pdf.default_overlap", 150)

    @property
    def default_chunk_mode(self) -> str:
        return self._yaml_get("pdf.default_chunk_mode", "sliding")

    @property
    def default_author(self) -> str:
        return self._yaml_get("pdf.default_author", "未知作者")

    @property
    def document_title_patterns(self) -> list:
        return self._yaml_get("pdf.document_title_patterns", [])

    @property
    def exclude_doc_keywords(self) -> list:
        return self._yaml_get("pdf.exclude_doc_keywords", [])

    @property
    def exclude_content_keywords(self) -> list:
        return self._yaml_get("pdf.exclude_content_keywords", [])

    @property
    def split_by_article_keywords(self) -> list:
        return self._yaml_get("pdf.split_by_article_keywords", [])

    @property
    def keep_as_whole_keywords(self) -> list:
        return self._yaml_get("pdf.keep_as_whole_keywords", [])

    @property
    def pdf_metadata(self) -> dict:
        return self._yaml_get("pdf.pdf_metadata", {})

    @property
    def chinese_number_mapping(self) -> dict:
        return self._yaml_get("chinese_number_mapping.mapping", {})

    @property
    def enable_dynamic_conversion(self) -> bool:
        return self._yaml_get("chinese_number_mapping.enable_dynamic_conversion", True)

    @property
    def enable_digits(self) -> bool:
        return self._yaml_get("chinese_number_mapping.enable_digits", True)

    @property
    def enable_units(self) -> bool:
        return self._yaml_get("chinese_number_mapping.enable_units", True)

    @property
    def qr_enable_colloquial(self) -> bool:
        return self._yaml_get("question_rewriter.enable_colloquial_to_formal", True)

    @property
    def qr_enable_redundancy(self) -> bool:
        return self._yaml_get("question_rewriter.enable_redundancy_removal", True)

    @property
    def qr_enable_synonym(self) -> bool:
        return self._yaml_get("question_rewriter.enable_synonym_expansion", True)

    @property
    def qr_colloquial_mappings(self) -> dict:
        return self._yaml_get("question_rewriter.colloquial_mappings", {})

    @property
    def qr_redundancy_patterns(self) -> list:
        return self._yaml_get("question_rewriter.redundancy_patterns", [])

    @property
    def qr_redundant_phrases(self) -> list:
        return self._yaml_get("question_rewriter.redundant_phrases", [])

    @property
    def qr_synonym_mappings(self) -> dict:
        return self._yaml_get("question_rewriter.synonym_mappings", {})

    def validate(self) -> None:
        if not self.llm_api_url:
            raise ValueError("llm_api_url is required")
        if not self.llm_api_key:
            raise ValueError("llm_api_key is required")


settings = Settings()
