# coding: utf-8

from agent_layer.config import Settings
from agent_layer.models import EmbeddingGateway
from agent_layer.retrieval.fusion import HybridFusion
from agent_layer.retrieval.store import VectorStore


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

    def search_article_exact(self, law_name: str, article_num: str, as_of_date: str | None = None) -> list[dict]:
        filters = [f'article_id == "{self._escape(article_num)}"']
        if law_name:
            filters.append(f'law_name == "{self._escape(law_name)}"')
        filters.append(self.build_policy_filter(as_of_date))
        return self.store.query(
            self.settings.policy_collection,
            " and ".join(filters),
            self.settings.top_k,
        )

    def _select_fusion_method(self, query: str) -> str:
        return self.settings.fusion_strategy

    @staticmethod
    def build_policy_filter(as_of_date: str | None = None, region: str | None = None) -> str:
        filters = []
        if as_of_date:
            escaped = HybridRetriever._escape(as_of_date)
            filters.append(f'effective_date <= "{escaped}" and (end_date == "" or end_date > "{escaped}")')
        else:
            filters.append('validity_status == "effective"')
        if region:
            escaped_region = HybridRetriever._escape(region)
            filters.append(f'region in ["national", "{escaped_region}"]')
        return " and ".join(filters)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
