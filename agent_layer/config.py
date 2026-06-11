# coding: utf-8
# @Author: Wang Qingkang

import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


AGENT_LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_LAYER_ROOT.parent

MODEL_CACHE_ROOT = AGENT_LAYER_ROOT / "data" / "model_cache"
os.environ.setdefault("HF_HOME", (MODEL_CACHE_ROOT / "hf_home").as_posix())
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", (MODEL_CACHE_ROOT / "hub").as_posix())
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", (MODEL_CACHE_ROOT / "sentence_transformers").as_posix())


class Settings(BaseSettings):
    """Agent layer runtime settings."""

    app_title: str = "Tender QA RAG Agent"
    app_version: str = "0.4"
    app_description: str = "LangChain-based tender and bidding regulation assistant."

    llm_api_key: str = ""
    llm_api_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1200
    llm_timeout_seconds: float = 60.0

    embedding_model: str = "moka-ai/m3e-base"
    chroma_persist_dir: str = "agent_layer/data/databases/chroma_db"
    default_collection: str = "regulations"

    top_k: int = 5
    vector_recall: int = 20
    fusion_strategy: str = "smart"
    summarize_max_chunks: int = 8
    summarize_chunk_length: int = 600
    low_score_threshold: float = 0.25

    weights_keyword_heavy: tuple[float, float] = (0.75, 0.25)
    weights_semantic_heavy: tuple[float, float] = (0.40, 0.60)
    weights_balanced: tuple[float, float] = (0.65, 0.35)

    boost_law_article: float = 0.08
    boost_article_start: float = 0.05
    boost_high_keyword: float = 0.06
    boost_medium_keyword: float = 0.04
    boost_low_keyword: float = 0.03
    boost_regulation_keyword: float = 0.05
    boost_multi_category: float = 0.02
    boost_max: float = 0.20

    penalty_min: float = -0.30
    semantic_law_article_penalty: float = 0.50
    min_final_score: float = 0.10
    high_norm_threshold: float = 0.80
    dense_no_boost_penalty: float = 0.12

    react_max_steps: int = 5
    react_temperature: float = 0.30
    stream_chunk_size: int = 32

    agent_search_top_k: int = 3
    agent_search_max_text_len: int = 500
    agent_fallback_top_k: int = 3
    agent_article_max_results: int = 2
    agent_article_max_text_len: int = 800

    enable_question_rewrite: bool = True
    unrelated_response: str = "抱歉，我只能回答招投标相关问题。"
    no_results_response: str = "抱歉，根据现有知识库未找到相关信息。"

    greeting_responses: dict[str, str] = {
        "你好": "您好！我是招投标智能助手，可以为您解答招标投标、政府采购相关的法规政策问题。请问有什么可以帮您？",
        "您好": "您好！请问有什么招投标方面的问题可以帮您？",
        "hi": "Hello！我是招投标智能助手，有什么可以帮您？",
        "hello": "Hello！请问您想了解招标投标的哪方面内容？",
        "在吗": "在的，请问有什么可以帮您？",
        "谢谢": "不客气，有问题随时问我。",
        "感谢": "不客气。",
        "再见": "再见！如有问题，随时回来咨询。",
    }
    greeting_keywords: list[str] = ["你好", "您好", "hi", "hello", "嗨", "在吗", "在不在", "有人吗"]
    thanks_keywords: list[str] = ["谢谢", "感谢", "thanks", "thank"]
    goodbye_keywords: list[str] = ["再见", "拜拜", "bye", "goodbye"]
    unrelated_keywords: list[str] = [
        "天气",
        "气温",
        "下雨",
        "股票",
        "基金",
        "理财",
        "美食",
        "餐厅",
        "电影",
        "娱乐",
        "明星",
        "游戏",
        "足球",
        "篮球",
    ]
    bidding_keywords: list[str] = ["招标", "投标", "采购", "围标", "串标", "中标", "标书", "标段", "评标", "开标"]

    tender_keywords: dict[str, list[str]] = {
        "high": ["招标编号", "项目编号", "标段", "分包", "招标项目编号", "采购项目编号"],
        "medium": ["资质要求", "业绩要求", "注册资本", "建造师", "评标办法", "限价", "招标控制价", "保证金", "投标保证金", "资格条件", "资质等级"],
        "low": ["投标人", "招标人", "开标时间", "截止时间", "递交截止", "投标截止"],
    }
    regulation_keywords: dict[str, list[str]] = {
        "high": ["民法典", "招标投标法", "政府采购法", "招标投标法实施条例"],
        "medium": ["管理办法", "指导意见", "通知", "规定"],
    }
    keyword_heavy_patterns: list[str] = [
        r"第\d+条",
        r"编号|标段|资质|建造师|注册资本",
        r"限价|保证金|资格条件",
    ]
    semantic_heavy_patterns: list[str] = [
        r"什么是|是什么|定义|解释|含义|概念|意思",
        r"如何|怎么|怎样|步骤|流程|操作|办理",
        r"背景|原因|目的|意义|解读|分析",
        r"区别|不同|对比|比较",
        r"串通|围标|陪标|挂靠",
        r"投标人|招标人|评标",
    ]
    penalty_patterns: list[tuple[str, float, str]] = [
        (r"^第[一二三四五六七八九十]+页$", -0.20, "页码"),
        (r"^\s*目录\s*$", -0.20, "目录"),
        (r"^[（(]?\d+[）)]?\s*$", -0.20, "纯数字"),
        (r"(版权所有|All Rights Reserved)", -0.15, "版权声明"),
    ]

    colloquial_mappings: dict[str, str] = {
        "咋": "怎么",
        "啥": "什么",
        "干嘛": "做什么",
        "咋样": "怎么样",
        "咋办": "怎么办",
        "搞": "进行",
        "弄": "处理",
        "有没有": "是否有",
        "有没": "是否有",
        "帮我看": "查询",
        "帮我查": "查询",
        "告诉我": "请说明",
        "讲一下": "请说明",
        "说说": "请说明",
        "想问问": "请问",
        "问一下": "请问",
        "多少": "什么",
        "哪条": "哪一条",
        "为啥": "为什么",
    }
    redundant_phrases: list[str] = [
        "我想问一下",
        "我想请问",
        "我想知道",
        "请问一下",
        "帮我查一下",
        "帮我看看",
        "我想了解一下",
        "麻烦问一下",
        "能不能告诉我",
        "可以告诉我",
    ]
    redundancy_patterns: list[str] = [
        r"(请问){2,}",
        r"(你好){2,}",
        r"(那个){2,}",
        r"(这个){2,}",
        r"就是+",
        r"那个",
        r"这个",
        r"然后",
    ]
    synonym_mappings: dict[str, str] = {
        "串标": "串通投标",
        "围标": "串通投标",
        "陪标": "串通投标",
        "挂靠": "借用资质",
        "借资质": "借用资质",
        "打分办法": "评标办法",
        "评分标准": "评标办法",
        "拦标价": "招标控制价",
        "最高限价": "招标控制价",
        "底价": "招标控制价",
        "保函": "投标保证金",
        "招投标法": "招标投标法",
        "招标法": "招标投标法",
        "投标法": "招标投标法",
        "采购法": "政府采购法",
        "招标方": "招标人",
        "投标方": "投标人",
        "采购方": "采购人",
        "供应商": "投标人",
        "业主": "招标人",
        "甲方": "招标人",
        "乙方": "投标人",
        "罚款": "处罚",
        "截标": "投标截止",
        "开标会": "开标",
        "评标会": "评标",
        "交标": "递交投标文件",
    }

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
    def chroma_persist_path(self) -> Path:
        return self._resolve_from_project(self.chroma_persist_dir)

    @property
    def embedding_model_name_or_path(self) -> str:
        cache_dir = MODEL_CACHE_ROOT / "sentence_transformers" / f"models--{self.embedding_model.replace('/', '--')}" / "snapshots"
        if cache_dir.exists():
            snapshots = sorted(path for path in cache_dir.iterdir() if path.is_dir())
            if snapshots:
                return snapshots[-1].as_posix()
        return self.embedding_model

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")


settings = Settings()


class LLMRequestError(Exception):
    """Raised when the configured chat model cannot be called."""


def build_chat_model(temperature: float = 0.3, max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.effective_llm_api_key or "missing-key",
        base_url=settings.llm_api_url,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )
