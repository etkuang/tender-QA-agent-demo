# coding: utf-8
# @Author: Wang Qingkang

import time

from agent_layer.application.services.react_agent_service import ReActAgent
from agent_layer.application.services.rewrite_service import rewrite_question
from agent_layer.domain.intent.router import IntentRouter
from agent_layer.domain.tools.regulation_tools import summarize
from agent_layer.infrastructure.config.agent_settings import settings
from agent_layer.infrastructure.llm.client import LLMGenerator, LLMRequestError
from agent_layer.domain.retrieval.retriever import HybridRetriever


class AskService:
    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator):
        self.retriever = retriever
        self.llm = llm
        self.intent_router = IntentRouter(llm)
        self.agent = ReActAgent(retriever, llm, max_steps=settings.react_max_steps)

    async def ask(self, question: str) -> tuple:
        start = time.time()
        rewritten = rewrite_question(question)
        intent = await self.intent_router.route(
            question=rewritten,
            session_id="",
            session_manager=None,
            history_context="",
        )

        route = "rag"
        sources = []

        if intent["type"] == "quick_response":
            return intent.get("response", "Hello, how can I help you?"), "greeting", time.time() - start, sources

        if intent["type"] == "unrelated":
            return settings.unrelated_response, "rejected", time.time() - start, sources

        if intent["complexity"] == "single_step":
            results = self.retriever.search(rewritten, "regulations", top_k=settings.top_k)
            sources = self._build_sources(results)
            if not results:
                return settings.no_results_response, route, time.time() - start, sources
            try:
                answer = await summarize(results, rewritten, self.llm)
            except LLMRequestError:
                answer = settings.no_results_response
            return answer, route, time.time() - start, sources

        answer = await self.agent.run(rewritten)
        return answer, "agent", time.time() - start, sources

    def _build_sources(self, results: list[dict]) -> list[dict]:
        output = []
        for item in results[: settings.top_k]:
            data = item.get("data", {})
            output.append(
                {
                    "title": data.get("doc_title", data.get("source", "unknown")),
                    "content_preview": item.get("text", "")[:200],
                    "score": item.get("score", 0),
                }
            )
        return output