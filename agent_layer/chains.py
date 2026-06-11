# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from common.logger import get_logger
from agent_layer.config import build_chat_model, settings
from agent_layer.prompts import INTENT_PROMPT, RAG_PROMPT
from agent_layer.retrieval import HybridRetriever, LegalRetrievalAdapter
from agent_layer.schemas import IntentDecision

logger = get_logger("agent.chains")


class IntentChain:
    def __init__(self):
        self.chain = INTENT_PROMPT | build_chat_model(temperature=0.0, max_tokens=300) | JsonOutputParser()

    async def ainvoke(self, question: str, history: str) -> IntentDecision:
        try:
            parsed = await self.chain.ainvoke({"question": question, "history": history})
        except Exception:
            logger.warning("intent_chain failed; falling back to single_step", exc_info=True)
            return IntentDecision(intent_type="other", complexity="single_step")

        intent_type = parsed.get("intent_type") or parsed.get("type") or "other"
        complexity = parsed.get("complexity") or "single_step"
        if intent_type not in {"definition", "procedure", "penalty", "provision", "other"}:
            intent_type = "other"
        if complexity not in {"single_step", "multi_step"}:
            complexity = "single_step"
        return IntentDecision(intent_type=intent_type, complexity=complexity)


class RagChain:
    def __init__(self, retriever: HybridRetriever):
        self.retrieval = LegalRetrievalAdapter(retriever)
        self.answer_chain = RAG_PROMPT | build_chat_model(temperature=settings.react_temperature) | StrOutputParser()

    async def ainvoke(self, question: str) -> dict:
        documents = self.retrieval.retrieve(question)
        if not documents:
            logger.info("rag_chain no documents")
            return {"answer": settings.no_results_response, "documents": [], "sources": []}

        context = self.retrieval.format_documents(documents)
        answer = await self.answer_chain.ainvoke({"question": question, "context": context})
        return {
            "answer": answer,
            "documents": documents,
            "sources": self.retrieval.build_sources(documents),
        }
