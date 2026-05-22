"""Agent 工具定义 - 提供给 ReAct Agent 调用的工具（配置化版本）"""
from typing import List, Dict, Any, Callable, Optional
from app.core.retriever import HybridRetriever
from config import settings


class BaseTool:
    """工具基类"""
    name: str = ""
    description: str = ""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    async def run(self, **kwargs) -> str:
        raise NotImplementedError


class SearchLawTool(BaseTool):
    """语义检索法规工具 - 用于模糊查询、概念定义、处罚规定等"""
    name = "search_law"
    description = """语义检索招投标法规知识库。
适用场景：问概念定义（什么是围标）、问处罚规定（串通投标罚款多少）、问流程步骤（如何开标）。
输入：自然语言问题，如"围标的定义"或"串通投标的处罚"
输出：相关法规片段（最多3条）"""

    async def run(self, query: str = "", **kwargs) -> str:
        if not query:
            return "错误：请提供检索关键词"

        # 从配置读取 top_k
        top_k = getattr(settings, 'agent_search_top_k', 3)
        results = self.retriever.search(query, "regulations", top_k=top_k)

        if not results:
            return f"未找到与「{query}」相关的信息"

        # 格式化输出
        output_parts = []
        max_text_len = getattr(settings, 'agent_search_max_text_len', 500)
        for i, r in enumerate(results[:top_k], 1):
            text = r.get("text", "")[:max_text_len]
            source = self._format_source(r)
            output_parts.append(f"[{i}] 来源：{source}\n{text}")

        return "\n\n".join(output_parts)

    def _format_source(self, chunk: Dict) -> str:
        data = chunk.get("data", {})
        doc_title = data.get("doc_title", "")
        article_num = data.get("article_num", "")

        if doc_title and doc_title != "unknown":
            clean_title = doc_title.replace("《", "").replace("》", "")
            if article_num and article_num not in ("unknown", "full"):
                return f"{clean_title} 第{article_num}条"
            return clean_title

        source = data.get("source", "未知来源")
        if article_num and article_num not in ("unknown", "full"):
            return f"{source} 第{article_num}条"
        return source


class GetArticleTool(BaseTool):
    """精确法条查询工具 - 用于查第X条的具体内容"""
    name = "get_article"
    description = """精确查询招投标法规的特定条款。
    适用场景：用户明确问到「第X条」的时候使用。
    输入格式：article_num（条款号，如"33"），可选 law_name（法律名称）
    示例输入：{"article_num": "XX"} 或 {"article_num": "XX", "law_name": "招标投标法"}
    输出：该条款的完整内容"""

    async def run(self, article_num: str = "", law_name: str = "", **kwargs) -> str:
        if not article_num:
            return "错误：请提供条款号（如 article_num=33）"

        # 调用精确检索方法
        results = self.retriever.search_article_exact(law_name, article_num)

        if not results:
            # 降级：尝试语义检索
            fallback_query = f"{law_name} 第{article_num}条" if law_name else f"第{article_num}条"
            top_k = getattr(settings, 'agent_fallback_top_k', 3)
            results = self.retriever.search(fallback_query, "regulations", top_k=top_k)
            if not results:
                return f"未找到第{article_num}条的相关内容"

        # 格式化输出
        output_parts = []
        max_results = getattr(settings, 'agent_article_max_results', 2)
        max_text_len = getattr(settings, 'agent_article_max_text_len', 800)
        for i, r in enumerate(results[:max_results], 1):
            text = r.get("text", "")[:max_text_len]
            source = self._format_source(r, article_num)
            output_parts.append(f"[{i}] {source}\n{text}")

        return "\n\n".join(output_parts)

    def _format_source(self, chunk: Dict, article_num: str) -> str:
        data = chunk.get("data", {})
        doc_title = data.get("doc_title", "")
        if doc_title and doc_title != "unknown":
            clean_title = doc_title.replace("《", "").replace("》", "")
            return f"来源：{clean_title} 第{article_num}条"
        source = data.get("source", "未知来源")
        return f"来源：{source} 第{article_num}条"


# 工具注册表
TOOL_CLASSES = {
    "search_law": SearchLawTool,
    "get_article": GetArticleTool,
}