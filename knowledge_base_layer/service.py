# coding: utf-8
# @Author: Wang Qingkang

from common.api_contracts.knowledge_base_api import KnowledgeHit, KnowledgeSearchRequest, KnowledgeSearchResponse
from knowledge_base_layer.config import Settings
from knowledge_base_layer.retrieval.parent_context import ParentContextExpander
from knowledge_base_layer.retrieval.reranker import Reranker
from knowledge_base_layer.retrieval.retriever import HybridRetriever


class KnowledgeBaseService:
    def __init__(
        self,
        retriever: HybridRetriever,
        parent_expander: ParentContextExpander,
        settings: Settings,
        reranker: Reranker | None = None,
    ):
        self.retriever = retriever
        self.parent_expander = parent_expander
        self.settings = settings
        self.reranker = reranker

    def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        collection = self.settings.policy_collection_alias
        as_of_date = request.as_of_date.isoformat() if request.as_of_date else None
        chunks = []
        if request.article_id:
            chunks = self.retriever.search_article_exact(
                collection,
                request.law_name or "",
                request.article_id,
                as_of_date,
                request.region,
                request.top_k * 2,
            )
        if not chunks:
            scalar_filter = self.retriever.build_policy_filter(as_of_date, request.region)
            chunks = self.retriever.search(
                request.query,
                collection,
                top_k=request.top_k * 2,
                scalar_filter=scalar_filter,
            )
        if self.settings.parent_context_enabled:
            chunks = self.parent_expander.expand(collection, chunks)
        if self.settings.reranker_enabled:
            if self.reranker is None:
                raise RuntimeError("Reranker is enabled but no implementation was injected.")
            chunks = self.reranker.rerank(
                request.query,
                chunks[: self.settings.reranker_candidate_pool],
                request.top_k,
            )
        hits = [
            KnowledgeHit(
                id=chunk["id"],
                text=chunk.get("text", ""),
                data=chunk.get("data", {}),
                score=chunk.get("score"),
                vector_rank=chunk.get("vector_rank"),
                bm25_rank=chunk.get("bm25_rank"),
                fusion_score=chunk.get("fusion_score"),
            )
            for chunk in chunks[: request.top_k]
        ]
        return KnowledgeSearchResponse(hits=hits)