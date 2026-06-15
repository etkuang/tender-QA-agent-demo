# coding: utf-8

import hashlib

from knowledge_base_layer.config import Settings


class HybridFusion:
    def __init__(self, settings: Settings):
        self.settings = settings

    def fuse(
        self,
        query: str,
        vector_results: list[dict],
        keyword_results: list[dict],
        strategy: str,
        top_k: int,
    ) -> list[dict]:
        if strategy == "weighted":
            return self._weighted_fusion(vector_results, keyword_results)[:top_k]
        return self._rrf_fusion(vector_results, keyword_results)[:top_k]

    def _weighted_fusion(self, vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        vector_scores = self._min_max_normalize([item.get("score", 0) for item in vector_results])
        keyword_scores = self._min_max_normalize([item.get("score", 0) for item in keyword_results])
        merged = self._merge_ranked_results(vector_results, keyword_results)
        for rank, item in enumerate(vector_results, 1):
            merged[self._document_id(item)]["vector_score"] = vector_scores[rank - 1]
        for rank, item in enumerate(keyword_results, 1):
            merged[self._document_id(item)]["bm25_score"] = keyword_scores[rank - 1]
        for item in merged.values():
            item["fusion_score"] = (
                self.settings.fusion_dense_weight * item["vector_score"]
                + self.settings.fusion_keyword_weight * item["bm25_score"]
            )
            item["score"] = item["fusion_score"]
        return sorted(merged.values(), key=lambda item: item["fusion_score"], reverse=True)

    def _rrf_fusion(self, vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        merged = self._merge_ranked_results(vector_results, keyword_results)
        for item in merged.values():
            if item["vector_rank"] is not None:
                item["fusion_score"] += 1 / (self.settings.fusion_rrf_k + item["vector_rank"])
            if item["bm25_rank"] is not None:
                item["fusion_score"] += 1 / (self.settings.fusion_rrf_k + item["bm25_rank"])
            item["score"] = item["fusion_score"]
        return sorted(merged.values(), key=lambda item: item["fusion_score"], reverse=True)

    def _merge_ranked_results(self, vector_results: list[dict], keyword_results: list[dict]) -> dict:
        merged = {}
        for rank, item in enumerate(vector_results, 1):
            document_id = self._document_id(item)
            merged.setdefault(document_id, self._base_item(item))
            merged[document_id]["vector_rank"] = rank
        for rank, item in enumerate(keyword_results, 1):
            document_id = self._document_id(item)
            merged.setdefault(document_id, self._base_item(item))
            merged[document_id]["bm25_rank"] = rank
        return merged

    @staticmethod
    def _base_item(item: dict) -> dict:
        return {
            "id": item.get("id", ""),
            "text": item.get("text", ""),
            "data": item.get("data", {}),
            "score": 0.0,
            "vector_rank": None,
            "bm25_rank": None,
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "fusion_score": 0.0,
        }

    @staticmethod
    def _document_id(item: dict) -> str:
        if item.get("id"):
            return item["id"]
        return hashlib.sha256(item.get("text", "").encode("utf-8")).hexdigest()

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [0.5] * len(scores)
        return [(score - minimum) / (maximum - minimum) for score in scores]
