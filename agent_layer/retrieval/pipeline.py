# coding: utf-8

import re
import time

from common.api_contracts.knowledge_base_api import KnowledgeSearchRequest
from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import PolicyRetrievalError
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.chinese_number import ChineseNumberConverter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.schemas import Evidence, PolicyQuery, ToolEvent

logger = get_logger("agent.retrieval.pipeline")


class RetrievalOutput:
    def __init__(self, evidence: list[Evidence], events: list[ToolEvent]):
        self.evidence = evidence
        self.events = events


class RetrievalPipeline:
    def __init__(
        self,
        client: KnowledgeBaseClient,
        adapter: EvidenceAdapter,
        settings: Settings,
    ):
        self.client = client
        self.adapter = adapter
        self.settings = settings
        self.number_converter = ChineseNumberConverter()

    async def retrieve_policy(
        self,
        question: str,
        policy_query: PolicyQuery | None = None,
    ) -> RetrievalOutput:
        started = time.perf_counter()
        events = [ToolEvent(stage="policy_retrieve", status="started", summary="开始检索政策知识库。")]
        try:
            parsed_query = policy_query or PolicyQuery()
            article_num = parsed_query.article_id or self.number_converter.extract_article_number(question)
            law_name = parsed_query.law_name or self._extract_law_name(question)
            response = await self.client.search(
                KnowledgeSearchRequest(
                    query=question,
                    top_k=self.settings.retrieval_batch_size,
                    law_name=law_name or None,
                    article_id=article_num,
                    as_of_date=parsed_query.as_of_date,
                    region=parsed_query.region,
                )
            )
            chunks = [hit.model_dump() for hit in response.hits]
            evidence = self._adapt_policy(chunks)
        except Exception as exc:
            logger.exception("policy knowledge-base retrieval failed")
            raise PolicyRetrievalError from exc
        events.append(
            ToolEvent(
                stage="policy_retrieve",
                status="completed",
                summary=f"政策知识库检索完成，确认 {len(evidence)} 条证据。",
                duration_ms=(time.perf_counter() - started) * 1000,
                details={
                    "query": question,
                    "article_exact": article_num not in (None, ""),
                    "law_name": law_name,
                    "as_of_date": parsed_query.as_of_date.isoformat() if parsed_query.as_of_date else None,
                    "region": parsed_query.region,
                },
            )
        )
        return RetrievalOutput(evidence, events)

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
            if len(output) >= self.settings.retrieval_batch_size:
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
