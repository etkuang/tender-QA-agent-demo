# coding: utf-8
# @Author: Wang Qingkang

import re
import time

from langchain.agents import create_agent

from agent_layer.chains import IntentChain, RagChain
from agent_layer.config import build_chat_model, settings
from agent_layer.prompts import LEGAL_AGENT_SYSTEM_PROMPT
from agent_layer.retrieval import HybridRetriever, QuestionRewriter
from agent_layer.schemas import AskResult
from agent_layer.tools import build_legal_tools


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            text_value = item.get("text")
            if isinstance(text_value, str):
                parts.append(text_value)
    return "".join(parts)


def normalize_question(question: str) -> str:
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", question.strip().lower())


def quick_response(question: str) -> str | None:
    normalized = normalize_question(question)

    for keyword in settings.greeting_keywords:
        if keyword in normalized:
            return settings.greeting_responses.get(keyword, "您好，我是招投标智能助手，请问有什么可以帮助您？")
    for keyword in settings.thanks_keywords:
        if keyword in normalized:
            return "不客气，有问题随时问我。"
    for keyword in settings.goodbye_keywords:
        if keyword in normalized:
            return "再见，如有问题，随时回来咨询。"
    if len(normalized) < 2:
        return "您好，请输入具体问题。"
    return None


def is_unrelated(question: str) -> bool:
    lower_question = question.lower()
    for keyword in settings.bidding_keywords:
        if keyword in lower_question:
            return False
    for keyword in settings.unrelated_keywords:
        if keyword in lower_question:
            return True
    return len(lower_question) < 5


class LegalAgent:
    def __init__(self, retriever: HybridRetriever):
        self.agent = create_agent(
            model=build_chat_model(temperature=settings.react_temperature),
            tools=build_legal_tools(retriever),
            system_prompt=LEGAL_AGENT_SYSTEM_PROMPT,
        )

    @staticmethod
    def _normalize_messages(history_messages: list[dict] | None, user_message: str) -> list[dict]:
        if not history_messages:
            return [{"role": "user", "content": user_message}]

        output = []
        for message in history_messages[-12:]:
            role = message.get("role")
            content = message.get("content", "")
            if role in {"system", "assistant", "user"} and isinstance(content, str) and content:
                output.append({"role": role, "content": content})

        if not output or output[-1]["role"] != "user":
            output.append({"role": "user", "content": user_message})
        else:
            output[-1]["content"] = user_message
        return output

    async def ainvoke(self, user_message: str, history_messages: list[dict] | None = None) -> str:
        result = await self.agent.ainvoke(
            {"messages": self._normalize_messages(history_messages, user_message)},
            config={"recursion_limit": settings.react_max_steps * 2 + 2},
        )
        final_message = result["messages"][-1]
        answer = extract_text(getattr(final_message, "content", ""))
        return answer if answer else settings.no_results_response


class TenderAgentRuntime:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.rewriter = QuestionRewriter()
        self.intent_chain = IntentChain()
        self.rag_chain = RagChain(self.retriever)
        self.agent = LegalAgent(self.retriever)

    @staticmethod
    def _history_context(history_messages: list[dict] | None) -> str:
        if not history_messages:
            return ""

        lines = []
        for message in history_messages[-6:]:
            role = message.get("role")
            content = message.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines)

    async def ask(self, user_message: str, history_messages: list[dict] | None = None) -> AskResult:
        start = time.time()
        rewritten_user_message = self.rewriter.rewrite(user_message)

        quick = quick_response(rewritten_user_message)
        if quick:
            return AskResult(answer=quick, route="greeting", processing_time=time.time() - start)

        if is_unrelated(rewritten_user_message):
            return AskResult(answer=settings.unrelated_response, route="rejected", processing_time=time.time() - start)

        intent = await self.intent_chain.ainvoke(rewritten_user_message, self._history_context(history_messages))
        if intent.complexity == "single_step":
            rag_result = await self.rag_chain.ainvoke(rewritten_user_message)
            return AskResult(
                answer=rag_result["answer"],
                route="rag",
                processing_time=time.time() - start,
                sources=rag_result["sources"],
            )

        answer = await self.agent.ainvoke(rewritten_user_message, history_messages)
        return AskResult(answer=answer, route="agent", processing_time=time.time() - start)
