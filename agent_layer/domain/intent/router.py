import json
import re
from dataclasses import dataclass

from agent_layer.infrastructure.llm.client import LLMGenerator, LLMRequestError
from agent_layer.infrastructure.config.agent_settings import settings


INTENT_PROMPT = """判断以下招标投标问题的类型和复杂度。

## 历史对话（仅最近一轮）
{history}

## 当前问题
{question}

只输出 JSON：
{{"type": "definition|procedure|penalty|provision|other", "complexity": "single_step|multi_step"}}
"""


@dataclass(slots=True)
class IntentDecision:
    intent_type: str
    complexity: str
    response: str | None = None

    def to_dict(self) -> dict:
        payload = {"type": self.intent_type, "complexity": self.complexity}
        if self.response:
            payload["response"] = self.response
        return payload


def _normalize_question(question: str) -> str:
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", question.strip().lower())


def _quick_response(question: str) -> str | None:
    normalized = _normalize_question(question)

    for keyword in settings.greeting_keywords:
        if keyword in normalized:
            return settings.greeting_responses.get(keyword, "您好！我是招投标智能助手，请问有什么可以帮您？")

    for keyword in settings.thanks_keywords:
        if keyword in normalized:
            return "不客气，有问题随时问我！"

    for keyword in settings.goodbye_keywords:
        if keyword in normalized:
            return "再见！如有问题，随时回来咨询。"

    if len(normalized) < 2:
        return "您好，请输入具体的问题。例如：什么是串通投标？"
    return None


def _is_unrelated(question: str) -> bool:
    lower_question = question.lower()
    for keyword in settings.bidding_keywords:
        if keyword in lower_question:
            return False

    for keyword in settings.unrelated_keywords:
        if keyword in lower_question:
            return True

    return len(lower_question) < 5


class IntentRouter:
    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def _history_context(self, session_id: str, session_manager, max_messages: int = 2) -> str:
        if not session_id or not session_manager:
            return "（无历史记录）"

        try:
            history = session_manager.get_chat_history(session_id)
            messages = history.messages[-max_messages:] if max_messages > 0 else history.messages
            if not messages:
                return "（无历史记录）"

            lines = []
            for message in messages:
                role = "用户" if message.type == "human" else "助手"
                lines.append(f"{role}: {message.content[:300]}")
            return "\n".join(lines)
        except Exception:
            return "（无历史记录）"

    async def route(
        self,
        question: str,
        session_id: str = "",
        session_manager=None,
        history_context: str = "",
    ) -> dict:
        quick = _quick_response(question)
        if quick:
            return IntentDecision("quick_response", "single_step", quick).to_dict()

        if _is_unrelated(question):
            return IntentDecision("unrelated", "single_step").to_dict()

        if not history_context:
            history_context = self._history_context(session_id, session_manager, max_messages=2)

        prompt = INTENT_PROMPT.format(history=history_context, question=question)
        try:
            response = await self.llm.generate(prompt, temperature=0.0)
            match = re.search(r"\{[^{}]*\}", response)
            if not match:
                return IntentDecision("other", "single_step").to_dict()

            parsed = json.loads(match.group())
            intent_type = parsed.get("type", "other")
            complexity = parsed.get("complexity", "single_step")
            return IntentDecision(intent_type, complexity).to_dict()
        except (json.JSONDecodeError, LLMRequestError, KeyError, TypeError):
            return IntentDecision("other", "single_step").to_dict()