from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
import os

from app.core.config_loader import config_loader


class Settings(BaseSettings):
    """应用配置"""

    # ========== LLM配置 ==========
    llm_api_key: str = ""
    llm_api_url: str = ""
    llm_model: str = "gpt-3.5-turbo"

    # ========== 数据路径 ==========
    pdf_dir: str = "./data/pdfs"
    chroma_persist_dir: str = "./data/databases/chroma_db"

    # ========== 检索配置 ==========
    embedding_model: str = "thenlper/gte-large-zh"
    top_k: int = 5
    vector_recall: int = 20

    # ========== 服务配置 ==========
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # 融合策略
    fusion_strategy: str = "smart"

    # 新方案权重配置
    bm25_weight_default: float = 0.65
    dense_weight_default: float = 0.35
    bm25_weight_keyword: float = 0.75
    dense_weight_keyword: float = 0.25
    bm25_weight_semantic: float = 0.40
    dense_weight_semantic: float = 0.60

    # ========== 只保留 model_config，删除 class Config ==========
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow"  # 允许额外字段
    }

    # ========== 以下所有 @property 保持不变 ==========

    @property
    def boost_law_article(self) -> float:
        return float(os.getenv("BOOST_LAW_ARTICLE", "0.08"))

    @property
    def boost_article_start(self) -> float:
        return float(os.getenv("BOOST_ARTICLE_START", "0.05"))

    @property
    def boost_high_keyword(self) -> float:
        return float(os.getenv("BOOST_HIGH_KEYWORD", "0.06"))

    @property
    def boost_medium_keyword(self) -> float:
        return float(os.getenv("BOOST_MEDIUM_KEYWORD", "0.04"))

    @property
    def boost_low_keyword(self) -> float:
        return float(os.getenv("BOOST_LOW_KEYWORD", "0.03"))

    @property
    def boost_regulation_keyword(self) -> float:
        return float(os.getenv("BOOST_REGULATION_KEYWORD", "0.05"))

    @property
    def boost_multi_category(self) -> float:
        return float(os.getenv("BOOST_MULTI_CATEGORY", "0.02"))

    @property
    def boost_max(self) -> float:
        return float(os.getenv("BOOST_MAX", "0.2"))

    @property
    def penalty_min(self) -> float:
        return float(os.getenv("PENALTY_MIN", "-0.3"))

    @property
    def semantic_law_article_penalty(self) -> float:
        return float(os.getenv("SEMANTIC_LAW_ARTICLE_PENALTY", "0.5"))

    @property
    def min_final_score(self) -> float:
        return float(os.getenv("MIN_FINAL_SCORE", "0.1"))

    @property
    def high_norm_threshold(self) -> float:
        return float(os.getenv("HIGH_NORM_THRESHOLD", "0.8"))

    @property
    def dense_no_boost_penalty(self) -> float:
        return float(os.getenv("DENSE_NO_BOOST_PENALTY", "0.12"))

    # ========== 动态权重配置（从YAML加载） ==========
    @property
    def weights_keyword_heavy(self) -> tuple:
        weights = config_loader.get("retrieval.weights.keyword_heavy", [0.75, 0.25])
        return (weights[0], weights[1])

    @property
    def weights_semantic_heavy(self) -> tuple:
        weights = config_loader.get("retrieval.weights.semantic_heavy", [0.40, 0.60])
        return (weights[0], weights[1])

    @property
    def weights_balanced(self) -> tuple:
        weights = config_loader.get("retrieval.weights.balanced", [0.65, 0.35])
        return (weights[0], weights[1])

    # ========== 关键词配置（从YAML加载） ==========
    @property
    def tender_keywords(self) -> dict:
        return config_loader.get("retrieval.boost_keywords", {"high": [], "medium": [], "low": []})

    @property
    def regulation_keywords(self) -> dict:
        return config_loader.get("retrieval.regulation_keywords", {"high": [], "medium": []})

    @property
    def penalty_patterns(self) -> list:
        patterns = config_loader.get("retrieval.penalty_patterns", [])
        return [(p[0], p[1], p[2]) for p in patterns]

    # ========== 意图分类配置（从YAML加载） ==========
    @property
    def greeting_responses(self) -> Dict[str, str]:
        return config_loader.get("intent.greeting_responses", {})

    @property
    def greeting_keywords(self) -> list:
        return config_loader.get("intent.greeting_keywords", [])

    @property
    def thanks_keywords(self) -> list:
        return config_loader.get("intent.thanks_keywords", [])

    @property
    def goodbye_keywords(self) -> list:
        return config_loader.get("intent.goodbye_keywords", [])

    @property
    def unrelated_keywords(self) -> list:
        return config_loader.get("intent.unrelated_keywords", [])

    @property
    def bidding_keywords(self) -> list:
        return config_loader.get("intent.bidding_keywords", [])

    @property
    def keyword_heavy_patterns(self) -> list:
        return config_loader.get("retrieval.keyword_heavy_patterns", [])

    @property
    def semantic_heavy_patterns(self) -> list:
        return config_loader.get("retrieval.semantic_heavy_patterns", [])

    # ========== PDF处理配置 ==========
    @property
    def default_chunk_size(self) -> int:
        return int(os.getenv("DEFAULT_CHUNK_SIZE", "800"))

    @property
    def default_overlap(self) -> int:
        return int(os.getenv("DEFAULT_OVERLAP", "150"))

    @property
    def default_chunk_mode(self) -> str:
        return os.getenv("DEFAULT_CHUNK_MODE", "sliding")

    @property
    def default_author(self) -> str:
        return os.getenv("DEFAULT_AUTHOR", "未知作者")

    @property
    def manual_eval_path(self) -> str:
        return os.getenv("MANUAL_EVAL_PATH", "./data/manual_eval_set.json")

    # ========== 文档标题匹配规则（从YAML加载） ==========
    @property
    def document_title_patterns(self) -> list:
        return config_loader.get("pdf.document_title_patterns", [
            r'^中华人民共和国',
            r'[法条例规定办法意见通知批复函]$',
            r'^关于.*[通知意见函]$'
        ])

    @property
    def exclude_doc_keywords(self) -> list:
        return config_loader.get("pdf.exclude_doc_keywords", [
            '目录', '编辑出版说明', '图书在版编目', 'CIP', 'ISBN',
            '总目录', '资料补充栏', '前言', '编写说明', '出版说明',
            '索引', '附录', '附件', '附表', '格式', '样板'
        ])

    @property
    def exclude_content_keywords(self) -> list:
        return config_loader.get("pdf.exclude_content_keywords", [
            '审计法', '预算法', '价格法', '公证法', '行政处罚法',
            '残疾人', '中小企业', '脱贫攻坚', '扶贫'
        ])

    @property
    def split_by_article_keywords(self) -> list:
        return config_loader.get("pdf.split_by_article_keywords", [
            '法', '条例', '规定', '办法', '规范', '细则'
        ])

    @property
    def keep_as_whole_keywords(self) -> list:
        return config_loader.get("pdf.keep_as_whole_keywords", [
            '通知', '意见', '批复', '函', '复函'
        ])

    @property
    def pdf_metadata(self) -> dict:
        return config_loader.get("pdf.pdf_metadata", {})

    # ========== 应用配置 ==========
    @property
    def app_title(self) -> str:
        return os.getenv("APP_TITLE", "招投标RAG+ReAct Agent")

    @property
    def app_version(self) -> str:
        return os.getenv("APP_VERSION", "2.1.0")

    @property
    def app_description(self) -> str:
        return os.getenv("APP_DESCRIPTION", "招投标法规智能问答系统")

    # ========== 响应消息配置 ==========
    @property
    def unrelated_response(self) -> str:
        return os.getenv("UNRELATED_RESPONSE", "抱歉，我只能回答招标投标相关的问题。")

    @property
    def no_results_response(self) -> str:
        return os.getenv("NO_RESULTS_RESPONSE", "抱歉，未找到相关信息。")

    # ========== ReAct Agent配置 ==========
    @property
    def react_max_steps(self) -> int:
        return int(os.getenv("REACT_MAX_STEPS", "5"))

    @property
    def react_temperature(self) -> float:
        return float(os.getenv("REACT_TEMPERATURE", "0.3"))

    # ========== Agent 工具配置 ==========
    @property
    def agent_search_top_k(self) -> int:
        return int(os.getenv("AGENT_SEARCH_TOP_K", "3"))

    @property
    def agent_search_max_text_len(self) -> int:
        return int(os.getenv("AGENT_SEARCH_MAX_TEXT_LEN", "500"))

    @property
    def agent_fallback_top_k(self) -> int:
        return int(os.getenv("AGENT_FALLBACK_TOP_K", "3"))

    @property
    def agent_article_max_results(self) -> int:
        return int(os.getenv("AGENT_ARTICLE_MAX_RESULTS", "2"))

    @property
    def agent_article_max_text_len(self) -> int:
        return int(os.getenv("AGENT_ARTICLE_MAX_TEXT_LEN", "800"))
    # ========== Summarize 配置 ==========
    @property
    def summarize_max_chunks(self) -> int:
        return int(os.getenv("SUMMARIZE_MAX_CHUNKS", "8"))

    @property
    def summarize_chunk_length(self) -> int:
        return int(os.getenv("SUMMARIZE_CHUNK_LENGTH", "600"))

    @property
    def low_score_threshold(self) -> float:
        return float(os.getenv("LOW_SCORE_THRESHOLD", "0.25"))

    # ========== 会话存储配置（SQLite） ==========
    sqlite_db_path: str = "./data/sessions.db"
    session_ttl: int = 3600
    # ==========会话管理配置==========
    session_ttl_seconds: int = 604800  # 会话 TTL（秒），7天
    max_messages_per_session: int = 30  # 每会话最大消息数
    cleanup_interval_seconds: int = 3600  # 清理检查间隔（秒），1小时
    # 否启用问题重写
    enable_question_rewrite: bool = True  # ← 添加这行，作为实例变量
    #===============generator对输出最大token设置=============
    @property
    def llm_max_tokens(self) -> int:
        return config_loader.get("llm.max_tokens", 1000)


    # ========== 中文数字转换配置（从YAML加载） ==========

    @property
    def chinese_number_mapping(self) -> dict:
        """中文数字映射表"""
        return config_loader.get("chinese_number_mapping.mapping", {})

    @property
    def enable_dynamic_conversion(self) -> bool:
        """是否启用动态中文数字转换"""
        return config_loader.get("chinese_number_mapping.enable_dynamic_conversion", True)

    @property
    def enable_digits(self) -> bool:
        """是否启用算法转换中的中文数字字符"""
        return config_loader.get("chinese_number_mapping.enable_digits", True)

    @property
    def enable_units(self) -> bool:
        """是否启用算法转换中的中文单位"""
        return config_loader.get("chinese_number_mapping.enable_units", True)

    # ========== 问题轻量级改写配置 ==========
    @property
    def qr_enable_colloquial(self) -> bool:
        return config_loader.get("question_rewriter.enable_colloquial_to_formal", True)

    @property
    def qr_enable_redundancy(self) -> bool:
        return config_loader.get("question_rewriter.enable_redundancy_removal", True)

    @property
    def qr_enable_synonym(self) -> bool:
        return config_loader.get("question_rewriter.enable_synonym_expansion", True)

    @property
    def qr_colloquial_mappings(self) -> dict:
        return config_loader.get("question_rewriter.colloquial_mappings", {})

    @property
    def qr_redundancy_patterns(self) -> list:
        return config_loader.get("question_rewriter.redundancy_patterns", [])

    @property
    def qr_redundant_phrases(self) -> list:
        return config_loader.get("question_rewriter.redundant_phrases", [])

    @property
    def qr_synonym_mappings(self) -> dict:
        return config_loader.get("question_rewriter.synonym_mappings", {})
# 确保配置加载器已加载
config_loader.load("config.yaml")

settings = Settings()