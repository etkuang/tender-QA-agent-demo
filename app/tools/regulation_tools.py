# app/tools/regulation_tools.py
# -*- coding: utf-8 -*-

"""三个工具：search_regulation / get_article / summarize（配置化版本）"""
from typing import List, Dict, Any
from app.core.retriever import HybridRetriever
from app.core.generator import LLMGenerator
from config import settings
import re

# ========== 导入配置化的中文数字转换器 ==========
from app.utils.chinese_number import (
    chinese_number_converter,
    chinese_to_arabic_dynamic,
    chinese_to_arabic,
    extract_article_number
)


# ========== 删除原有的 chinese_to_arabic_dynamic、chinese_to_arabic、extract_article_number 函数 ==========
# 改为从 app.utils.chinese_number 导入


# ========== 工具1: 语义检索 ==========
async def search_regulation(query: str, retriever: HybridRetriever, top_k: int = None) -> List[Dict]:
    """语义检索法规内容"""
    if top_k is None:
        top_k = settings.top_k
    return retriever.search(query, "regulations", top_k)


# ========== 工具2: 精确查第X条 ==========
async def get_article(article_no: str, law_name: str, retriever: HybridRetriever) -> Dict:
    """精确查找某法第X条"""
    import re

    # 提取数字（使用配置化的转换器）
    article_num = extract_article_number(f"第{article_no}条") if "条" not in article_no else extract_article_number(
        article_no)

    # 降级：直接提取数字
    if not article_num:
        numbers = re.findall(r'\d+', article_no)
        article_num = numbers[0] if numbers else ""

    if not article_num:
        return {"success": False, "error": f"无法解析条款号: {article_no}"}

    # 从配置读取 top_k
    top_k = getattr(settings, 'agent_fallback_top_k', 3)

    # 尝试用元数据过滤精确匹配
    try:
        metadata_filter = {
            "article_num": article_num,
            "type": "law_article"
        }

        if law_name:
            metadata_filter["source"] = law_name

        results = retriever.search_with_filter(
            query=f"{law_name} 第{article_num}条",
            collection="regulations",
            metadata_filter=metadata_filter,
            top_k=top_k
        )

        if results:
            return {
                "success": True,
                "article": article_no,
                "law": law_name,
                "content": results[0]["text"],
                "source": results[0].get("data", {})
            }
    except Exception as e:
        print(f"精确匹配失败: {e}")

    # 降级到语义检索（使用阿拉伯数字）
    fallback = retriever.search(f"{law_name} 第{article_num}条", "regulations", top_k)
    if fallback:
        return {
            "success": True,
            "article": article_num,
            "law": law_name,
            "content": fallback[0]["text"],
            "source": fallback[0].get("data", {}),
            "note": "精确匹配未找到，此为语义检索结果"
        }

    return {"success": False, "error": f"未找到 {law_name} 第{article_num}条"}


# ========== 工具3: 多段归纳 ==========
SUMMARIZE_PROMPT = """你是一个招标投标领域的专家助手。请严格基于以下检索结果回答问题。

用户问题：{question}

检索结果（可能来自多轮检索，请综合所有信息）：
{chunks}

【核心规则 - 必须遵守】：
1. 【最重要】如果检索结果为空、完全不相关，或者没有任何一条结果能够支撑你的回答，请**只回复**："抱歉，根据现有知识库未找到相关信息。"
2. 【重要】请综合所有检索结果，不要只使用最后几条。如果多条结果包含不同维度的信息，都要整合到答案中。
3.  回答**只能**基于上述检索结果中明确写出的内容，**严禁使用你自身的知识或常识进行补充**。
4.  不要编造任何检索结果中没有提到的法条、处罚金额、定义等
5.  如果信息不完整，只回答有依据的部分，并说明"其他信息暂未找到"。
6.  每条信息后必须标注来源，格式：📚 来源：XXX（如：《招标投标法》第XX条）
7.  如果无法标注来源，说明信息不可靠。
8.  如果你不确定检索结果是否相关，宁可回答"未找到"，也不要猜测。

【示例】：
用户：公开招标和邀请招标的区别
检索结果：
[1] 来源：招标投标法第10条
公开招标是指以招标公告方式邀请不特定的法人投标...
[2] 来源：招标投标法律解读
公开招标竞争范围广，邀请招标竞争范围有限...
[3] 来源：招标投标法律解读
邀请招标程序比公开招标简化...
回答：
公开招标和邀请招标的区别如下：
1. **邀请对象**：公开招标邀请不特定的法人，邀请招标邀请特定的法人。📚 来源：《招标投标法》第10条
2. **竞争范围**：公开招标竞争范围广，邀请招标竞争范围有限。📚 来源：招标投标法律解读
3. **程序简化**：邀请招标程序比公开招标简化。📚 来源：招标投标法律解读
【错误示例 - 禁止】：
- 用户问"罚款多少？"，检索结果中没写 → ❌ "一般罚款5-10万元"（这是编造）
- 用户问"第33条"，检索结果中是第53条 → ❌ "第33条规定..."（这是错误的）

【正确示例】：
- 检索结果为空 → "抱歉，根据现有知识库未找到相关信息。"
- 检索结果只有定义，没有处罚 → "根据检索结果，...（定义部分）。关于处罚的其他信息暂未找到。"

请严格遵守以上规则。

"""


def _format_source(chunk: Dict) -> str:
    """
    格式化来源信息，优先使用 doc_title，其次使用 source + article_num
    """
    data = chunk.get("data", {})

    # 优先使用 doc_title（文档标题）
    doc_title = data.get("doc_title", "")
    if doc_title and doc_title != "unknown":
        # 如果有法条编号且不是"full"，加上第X条
        article_num = data.get("article_num", "")
        if article_num and article_num != "unknown" and article_num != "full":
            clean_title = doc_title.replace("《", "").replace("》", "")
            return f"{clean_title} 第{article_num}条"
        clean_title = doc_title.replace("《", "").replace("》", "")
        return clean_title

    # 降级：使用 source + article_num
    source = data.get("source", "未知来源")
    article_num = data.get("article_num", "")
    if article_num and article_num != "unknown" and article_num != "full":
        return f"{source} 第{article_num}条"

    return source


async def summarize(chunks: List[Dict], question: str, llm: LLMGenerator) -> str:
    """将多段检索结果归纳成结构化回答（配置化版本）"""

    # 空结果快速返回
    if not chunks:
        return settings.no_results_response

    print(f"   📊 summarize 收到 {len(chunks)} 条结果")

    # 从配置读取参数
    max_chunks = getattr(settings, 'summarize_max_chunks', 8)
    max_chunk_length = getattr(settings, 'summarize_chunk_length', 600)
    low_score_threshold = getattr(settings, 'low_score_threshold', 0.25)

    unique_chunks = chunks

    # 低分结果警告
    top_score = chunks[0].get("score", 0)
    low_quality_warning = ""
    if top_score < low_score_threshold:
        low_quality_warning = "\n\n⚠️ 注意：以下检索结果相关性较低，请谨慎回答，优先回复找不到信息。"

    # 格式化
    formatted = []
    for i, chunk in enumerate(unique_chunks[:max_chunks]):
        text = chunk.get("text", "")[:max_chunk_length]
        source_str = _format_source(chunk)
        formatted.append(f"[{i + 1}] 来源：{source_str}\n{text}")

    prompt = SUMMARIZE_PROMPT.format(
        question=question,
        chunks="\n\n".join(formatted) + low_quality_warning
    )

    answer = await llm.generate(prompt, temperature=settings.react_temperature)

    # 后处理检测幻觉
    if len(answer) > 50 and "📚 来源" not in answer and "未找到" not in answer:
        print(f"  ⚠️ 可能产生幻觉，答案无来源标注")

    return answer