# coding: utf-8

import re
import time

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import PolicyRetrievalError
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.chinese_number import ChineseNumberConverter
from agent_layer.retrieval.parent_context import ParentContextExpander
from agent_layer.retrieval.reranker import Reranker
from agent_layer.retrieval.retriever import HybridRetriever
from agent_layer.schemas import Category, Evidence, PolicyQuery, ToolEvent

logger = get_logger("agent.retrieval.pipeline")


class RetrievalOutput:
    def __init__(self, evidence: list[Evidence], events: list[ToolEvent]):
        self.evidence = evidence
        self.events = events


class RetrievalPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        adapter: EvidenceAdapter,
        settings: Settings,
        parent_expander: ParentContextExpander | None = None,
        reranker: Reranker | None = None,
    ):
        self.retriever = retriever
        self.adapter = adapter
        self.settings = settings
        self.number_converter = ChineseNumberConverter()
        self.parent_expander = parent_expander
        self.reranker = reranker

    def retrieve_policy(self, question: str, policy_query: PolicyQuery | None = None) -> RetrievalOutput:
        started = time.perf_counter()
        events = [ToolEvent(stage="policy_retrieve", status="started", summary="开始检索本地政策库。")]
        try:
            parsed_query = policy_query or PolicyQuery()
            article_num = parsed_query.article_id or self.number_converter.extract_article_number(question)
            law_name = parsed_query.law_name or self._extract_law_name(question)
            as_of_date = parsed_query.as_of_date.isoformat() if parsed_query.as_of_date else None
            chunks = []
            if article_num:
                chunks = self.retriever.search_article_exact(law_name, article_num, as_of_date)
            if not chunks:
                chunks = self.retriever.search(
                    question,
                    self.settings.policy_collection,
                    top_k=self.settings.top_k * 2,
                    scalar_filter=self.retriever.build_policy_filter(as_of_date, parsed_query.region),
                )
            expanded = chunks
            if self.settings.parent_context_enabled and self.parent_expander is not None:
                expanded = self.parent_expander.expand(self.settings.policy_collection, chunks)
            if self.settings.reranker_enabled:
                if self.reranker is None:
                    raise RuntimeError("Reranker is enabled but no implementation was injected.")
                expanded = self.reranker.rerank(
                    question,
                    expanded[: self.settings.reranker_candidate_pool],
                    self.settings.top_k,
                )
            evidence = self._adapt_policy(expanded)
        except Exception as exc:
            logger.exception("policy retrieval failed")
            raise PolicyRetrievalError from exc
        events.append(
            ToolEvent(
                stage="policy_retrieve",
                status="completed",
                summary=f"本地政策检索完成，确认 {len(evidence)} 条证据。",
                duration_ms=(time.perf_counter() - started) * 1000,
                details={
                    "article_exact": bool(article_num),
                    "law_name": law_name,
                    "as_of_date": as_of_date,
                    "region": parsed_query.region,
                },
            )
        )
        return RetrievalOutput(evidence, events)

    def retrieve_domain(self, question: str, category: Category, collection: str) -> RetrievalOutput:
        started = time.perf_counter()
        chunks = self.retriever.search(question, collection, top_k=self.settings.top_k)
        evidence = [self.adapter.from_domain_chunk(chunk, category) for chunk in chunks]
        event = ToolEvent(
            stage="local_domain_retrieve",
            status="completed",
            summary=f"本地历史索引检索完成，确认 {len(evidence)} 条记录。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"collection": collection},
        )
        return RetrievalOutput(evidence, [event])

    def _adapt_policy(self, chunks: list[dict]) -> list[Evidence]:
        output = []
        seen = set()
        for chunk in chunks:
            evidence = self.adapter.from_policy_chunk(chunk)
            key = (evidence.law_name, evidence.article_id, evidence.document_id)
            if key in seen:
                continue
            seen.add(key)
            output.append(evidence)
            if len(output) >= self.settings.top_k:
                break
        return output

    @staticmethod
    def _extract_law_name(question: str) -> str:
        match = re.search(r"《([^》]+)》", question)
        if match:
            return match.group(1)
        for law_name in ["中华人民共和国招标投标法实施条例", "中华人民共和国招标投标法", "中华人民共和国政府采购法"]:
            if law_name in question:
                return law_name
            short_name = law_name.replace("中华人民共和国", "")
            if short_name in question:
                return law_name
        return ""
