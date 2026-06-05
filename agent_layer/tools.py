# coding: utf-8

from langchain_core.tools import StructuredTool

from agent_layer.config import settings
from agent_layer.retrieval import HybridRetriever


def format_source(chunk: dict, fallback_article_num: str = "") -> str:
    metadata = chunk.get("data", {})
    title = metadata.get("doc_title") or metadata.get("law_name") or metadata.get("source") or "未知来源"
    article_num = metadata.get("article_num") or metadata.get("article_id") or fallback_article_num
    if article_num and article_num not in ("unknown", "full"):
        return f"{title} 第{article_num}条"
    return title


def build_legal_tools(retriever: HybridRetriever) -> list[StructuredTool]:
    async def search_law(query: str) -> str:
        """Search the tender and bidding regulation knowledge base."""
        results = retriever.search(query, settings.default_collection, top_k=settings.agent_search_top_k)
        if not results:
            return f"未找到与「{query}」相关的信息"

        output = []
        for index, item in enumerate(results[: settings.agent_search_top_k], 1):
            text = item.get("text", "")[: settings.agent_search_max_text_len]
            output.append(f"[{index}] 来源：{format_source(item)}\n{text}")
        return "\n\n".join(output)

    async def get_article(article_num: str, law_name: str = "") -> str:
        """Get the full text of a specific legal article."""
        results = retriever.search_article_exact(law_name, article_num)
        if not results:
            fallback_query = f"{law_name} 第{article_num}条" if law_name else f"第{article_num}条"
            results = retriever.search(fallback_query, settings.default_collection, top_k=settings.agent_fallback_top_k)
        if not results:
            return f"未找到第{article_num}条的相关内容"

        output = []
        for index, item in enumerate(results[: settings.agent_article_max_results], 1):
            text = item.get("text", "")[: settings.agent_article_max_text_len]
            output.append(f"[{index}] 来源：{format_source(item, article_num)}\n{text}")
        return "\n\n".join(output)

    return [
        StructuredTool.from_function(
            coroutine=search_law,
            name="search_law",
            description=(
                "语义检索招投标法规知识库。适用于概念定义、处罚规定、流程步骤、比较分析等问题。"
                "输入自然语言 query，输出相关法规片段和来源。"
            ),
        ),
        StructuredTool.from_function(
            coroutine=get_article,
            name="get_article",
            description=(
                "精确查询招投标法规的特定条款。用户明确问到第X条时使用。"
                "参数 article_num 为条款号，law_name 为可选法律名称。"
            ),
        ),
    ]
