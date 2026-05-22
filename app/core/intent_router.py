# app/core/intent_router.py
"""LLM意图分类器 - 判断问题类型和复杂度（配置化版本，支持历史感知）"""
import json
import re
from typing import Dict, Optional

from app.core.generator import LLMGenerator
from config import settings

# 增强版 Prompt，支持历史对话
INTENT_PROMPT = """判断以下招标投标问题的类型和复杂度。

## 历史对话（仅最近一轮，用于理解指代）
{history}

## 当前问题
{question}

输出JSON格式，不要有其他内容：
{{"type": "definition|procedure|penalty|provision|other", "complexity": "single_step|multi_step"}}

类型说明：
- definition: 问概念定义（什么是、如何理解）
- procedure: 问流程程序（如何操作、步骤）
- penalty: 问处罚后果（违反会怎样、罚款多少）
- provision: 问具体法条（第X条、规定）
- other: 其他

复杂度说明：
- single_step: 一次检索就能回答（如单条法条查询、简单定义）
- multi_step: 需要多次检索才能回答（如定义+处罚组合、对比分析）

示例：
问题："什么是串通投标？" → {{"type": "definition", "complexity": "single_step"}}
问题："围标的定义是什么，以及对应的处罚是多少？" → {{"type": "definition", "complexity": "multi_step"}}
问题："招标投标法第33条是什么？" → {{"type": "provision", "complexity": "single_step"}}

注意：如果当前问题使用了指代词（如"那"、"这个"、"上面"），请结合历史对话理解问题所指。
"""


def get_quick_response(question: str) -> Optional[str]:
    """快速匹配问候/致谢/告别，返回固定回复"""
    q = question.strip().lower()
    # 去除非中文字符（保留中文、字母、数字）
    q_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', q)

    for kw in settings.greeting_keywords:
        if kw in q_clean:  # 只保留这个条件，删除 kw == q
            default_response = "您好！我是招投标智能助手，请问有什么可以帮您？"
            return settings.greeting_responses.get(kw, default_response)

    for kw in settings.thanks_keywords:
        if kw in q_clean:
            return "不客气，有问题随时问我！"

    for kw in settings.goodbye_keywords:
        if kw in q_clean:
            return "再见！如有问题，随时回来咨询。"

    if len(q) < 2 or all(c in "？?！!。，,、；;：: " for c in q):
        return "您好，请输入具体的问题。例如：“什么是串通投标？”或“招标投标法第33条规定了什么？”"

    return None


def is_likely_unrelated(question: str) -> bool:
    """快速判断是否可能为无关问题"""
    q = question.lower()

    # 先检查是否包含招标关键词（包含则放行）
    for kw in settings.bidding_keywords:
        if kw in q:
            return False

    # 再检查无关关键词
    for kw in settings.unrelated_keywords:
        if kw in q:
            return True

    # 短查询且不包含招标关键词，可能是无关
    if len(q) < 5:
        return True

    # 移除原来的问号判断逻辑
    return False


class IntentRouter:
    """意图路由器 - 支持历史感知"""

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def _get_history_context(self, session_id: str, session_manager, max_messages: int = 2) -> str:
        """获取历史对话上下文（只取最近 N 条消息，默认2条=1轮）"""
        if not session_id or not session_manager:
            return "（无历史记录）"

        try:
            history = session_manager.get_chat_history(session_id)
            messages = history.messages[-max_messages:] if max_messages > 0 else history.messages

            if not messages:
                return "（无历史记录）"

            context_lines = []
            for msg in messages:
                role = "用户" if msg.type == "human" else "助手"
                content = msg.content[:300]  # 截断，意图路由不需要太长
                context_lines.append(f"{role}: {content}")

            return "\n".join(context_lines)
        except Exception as e:
            print(f"  ⚠️ 获取历史上下文失败: {e}")
            return "（无历史记录）"

    async def route(self, question: str, session_id: str = "", session_manager=None) -> Dict:
        """
        返回意图分类结果

        返回格式：
        {
            "type": "definition|procedure|penalty|provision|other|quick_response|unrelated",
            "complexity": "single_step|multi_step",
            "response": "..."  # 仅当 type="quick_response" 时存在
        }
        """

        # ========== 第一步：快速拦截（不需要历史） ==========
        quick_response = get_quick_response(question)
        if quick_response:
            return {
                "type": "quick_response",
                "complexity": "single_step",
                "response": quick_response
            }

        if is_likely_unrelated(question):
            print(f"🔍 [拦截] 无关问题: {question[:30]}...")
            return {"type": "unrelated", "complexity": "single_step"}

        # ========== 第二步：获取历史上下文（只取 1 轮） ==========
        history_context = self._get_history_context(session_id, session_manager, max_messages=2)

        # ========== 第三步：LLM 意图分类 ==========
        prompt = INTENT_PROMPT.format(
            history=history_context,
            question=question
        )

        try:
            response = await self.llm.generate(prompt, temperature=0)
            print(f"意图分类响应: {response[:200]}")

            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "type": result.get("type", "other"),
                    "complexity": result.get("complexity", "single_step")
                }
        except Exception as e:
            print(f"Intent routing error: {e}")

        return {"type": "other", "complexity": "single_step"}