# coding: utf-8

import re

import numpy as np

from agent_layer.config import settings
from agent_layer.retrieval.bm25 import BM25Cache
from agent_layer.retrieval.chroma import ChromaStore


class HybridFusion:
    def __init__(self, bm25_cache: BM25Cache | None = None):
        self.chroma_store = ChromaStore()
        self.bm25_cache = bm25_cache if bm25_cache else BM25Cache()

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> int:
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text or keyword.lower() in text_lower)

    def _compute_boost_score(self, text: str, metadata: dict) -> float:
        boost = 0.0
        hit_categories = set()

        if metadata.get("type") == "law_article" or metadata.get("chunk_type") == "parent":
            boost += settings.boost_law_article
            hit_categories.add("law_article")
        if re.match(r"^第\d+条", text.strip()):
            boost += settings.boost_article_start

        if self._contains_any(text, settings.tender_keywords.get("high", [])) >= 1:
            boost += settings.boost_high_keyword
            hit_categories.add("high")
        if self._contains_any(text, settings.tender_keywords.get("medium", [])) >= 2:
            boost += settings.boost_medium_keyword
            hit_categories.add("medium")
        if self._contains_any(text, settings.tender_keywords.get("low", [])) >= 2:
            boost += settings.boost_low_keyword
            hit_categories.add("low")
        if self._contains_any(text, settings.regulation_keywords.get("high", [])) >= 1:
            boost += settings.boost_regulation_keyword
            hit_categories.add("regulation")
        if len(hit_categories) >= 2:
            boost += settings.boost_multi_category

        return min(boost, settings.boost_max)

    @staticmethod
    def _compute_penalty_score(text: str) -> float:
        for pattern, penalty, reason in settings.penalty_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return max(penalty, settings.penalty_min)
        return 0.0

    @staticmethod
    def _detect_query_type(query: str) -> str:
        query_lower = query.lower()
        for pattern in settings.keyword_heavy_patterns:
            if re.search(pattern, query_lower):
                return "keyword_heavy"
        for pattern in settings.semantic_heavy_patterns:
            if re.search(pattern, query_lower):
                return "semantic_heavy"
        return "balanced"

    def _get_dynamic_weights(self, query: str) -> tuple[float, float]:
        query_type = self._detect_query_type(query)
        if query_type == "keyword_heavy":
            return settings.weights_keyword_heavy
        if query_type == "semantic_heavy":
            return settings.weights_semantic_heavy
        return settings.weights_balanced

    def search(self, query: str, collection: str, top_k: int) -> list[dict]:
        dense_results = self.chroma_store.search(collection, query, top_k=settings.vector_recall)
        if not dense_results:
            return []

        all_docs = self.chroma_store.get_all_documents(collection)
        if not all_docs:
            return dense_results[:top_k]

        keyword_results = self._keyword_search(query, collection, all_docs)
        self._attach_normalized_scores(dense_results, keyword_results)

        merged = self._merge_results(query, dense_results, keyword_results)
        output = self._filter_results(merged.values())
        return sorted(output, key=lambda item: item["final_score"], reverse=True)[:top_k]

    def _keyword_search(self, query: str, collection: str, all_docs: list[dict]) -> list[dict]:
        texts = [doc["text"] for doc in all_docs]
        bm25 = self.bm25_cache.get_or_build(collection, texts)
        bm25_scores = bm25.get_scores(BM25Cache.tokenize(query))
        top_indices = np.argsort(bm25_scores)[-settings.vector_recall:][::-1]

        output = []
        for index in top_indices:
            if bm25_scores[index] <= 0:
                continue
            output.append(
                {
                    "id": all_docs[index]["id"],
                    "score": float(bm25_scores[index]),
                    "data": all_docs[index]["metadata"],
                    "text": all_docs[index]["text"],
                }
            )
        return output

    def _attach_normalized_scores(self, dense_results: list[dict], keyword_results: list[dict]) -> None:
        dense_scores = self._min_max_normalize([item.get("score", 0) for item in dense_results])
        keyword_scores = self._min_max_normalize([item.get("score", 0) for item in keyword_results])

        for index, item in enumerate(dense_results):
            item["norm_score"] = dense_scores[index] if index < len(dense_scores) else 0
        for index, item in enumerate(keyword_results):
            item["norm_score"] = keyword_scores[index] if index < len(keyword_scores) else 0

    def _merge_results(
        self,
        query: str,
        dense_results: list[dict],
        keyword_results: list[dict],
    ) -> dict:
        bm25_weight, dense_weight = self._get_dynamic_weights(query)
        query_type = self._detect_query_type(query)
        merged = {}

        for source, items, weight in (("dense", dense_results, dense_weight), ("bm25", keyword_results, bm25_weight)):
            for item in items:
                scored_item = self._score_item(item, source, weight, query_type)
                doc_id = scored_item["id"]
                if doc_id not in merged or scored_item["final_score"] > merged[doc_id]["final_score"]:
                    merged[doc_id] = scored_item
        return merged

    def _score_item(self, item: dict, source: str, weight: float, query_type: str) -> dict:
        doc_id = item.get("id") or f"doc_{hash(item.get('text', ''))}"
        metadata = item.get("data", {})
        base_score = weight * item.get("norm_score", 0)

        if self._is_law_article(metadata) and query_type == "semantic_heavy":
            base_score *= settings.semantic_law_article_penalty

        boost = self._compute_boost_score(item.get("text", ""), metadata)
        penalty = self._compute_penalty_score(item.get("text", ""))
        final_score = base_score + boost + penalty

        return {
            "id": doc_id,
            "text": item.get("text", ""),
            "data": metadata,
            "score": final_score,
            "original_score": item.get("score", 0),
            "boost": boost,
            "penalty": penalty,
            "source": source,
            "final_score": final_score,
        }

    @staticmethod
    def _is_law_article(metadata: dict) -> bool:
        return metadata.get("type") == "law_article" or metadata.get("chunk_type") == "parent"

    @staticmethod
    def _filter_results(results) -> list[dict]:
        output = []
        for item in results:
            if item["final_score"] < settings.min_final_score:
                continue
            if item["source"] == "dense" and item["boost"] == 0 and item.get("original_score", 0) > settings.high_norm_threshold:
                item["final_score"] -= settings.dense_no_boost_penalty
                item["score"] = item["final_score"]
            output.append(item)
        return output
