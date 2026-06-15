# coding: utf-8

from knowledge_base_layer.config import Settings
from knowledge_base_layer.embeddings import EmbeddingGateway
from knowledge_base_layer.retrieval.fusion import HybridFusion
from knowledge_base_layer.retrieval.store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingGateway,
        fusion: HybridFusion,
        settings: Settings,
    ):
        self.store = store
        self.embeddings = embeddings
        self.fusion = fusion
        self.settings = settings

    def search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        scalar_filter: str = "",
    ) -> list[dict]:
        limit = top_k if top_k is not None else self.settings.top_k
        query_vector = self.embeddings.embed_query(query)
        vector_results = self.store.search_dense(
            collection,
            query_vector,
            self.settings.vector_recall,
            scalar_filter,
        )
        keyword_results = self.store.search_keyword(
            collection,
            query,
            self.settings.bm25_recall,
            scalar_filter,
        )
        return self.fusion.fuse(
            query,
            vector_results,
            keyword_results,
            self._select_fusion_method(query),
            limit,
        )

    def search_article_exact(
        self,
        collection: str,
        law_name: str,
        article_num: str,
        as_of_date: str | None,
        region: str | None,
        top_k: int,
    ) -> list[dict]:
        filters = [f'article_id == "{self._escape(article_num)}"']
        if law_name:
            filters.append(f'law_name == "{self._escape(law_name)}"')
        policy_filter = self.build_policy_filter(as_of_date, region)
        if policy_filter:
            filters.append(policy_filter)
        return self.store.query(collection, " and ".join(filters), top_k)

    def _select_fusion_method(self, query: str) -> str:
        return self.settings.fusion_strategy

    @staticmethod
    def build_policy_filter(as_of_date: str | None = None, region: str | None = None) -> str:
        filters = []
        if as_of_date:
            escaped = HybridRetriever._escape(as_of_date)
            filters.append(f'source_as_of <= "{escaped}"')
        if region:
            escaped_region = HybridRetriever._escape(region)
            filters.append(f'region in ["national", "{escaped_region}"]')
        return " and ".join(filters)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
