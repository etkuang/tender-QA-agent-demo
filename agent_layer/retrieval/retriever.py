# coding: utf-8
# @Author: Wang Qingkang

import re

import numpy as np

from agent_layer.config import settings
from agent_layer.retrieval.bm25 import BM25Cache
from agent_layer.retrieval.chinese_number import ChineseNumberConverter
from agent_layer.retrieval.chroma import ChromaStore
from agent_layer.retrieval.fusion import HybridFusion


class HybridRetriever:
    def __init__(self):
        self.chroma_store = ChromaStore()
        self.bm25_cache = BM25Cache()
        self.weighted_fusion = HybridFusion(self.bm25_cache)
        self.chinese_number_converter = ChineseNumberConverter()

    def search(self, query: str, collection: str | None = None, top_k: int | None = None) -> list[dict]:
        collection_name = collection or settings.default_collection
        limit = top_k if top_k is not None else settings.top_k
        article_num = self.chinese_number_converter.extract_article_number(query)

        if article_num and len(query) < 30:
            exact_results = self.search_article_exact("", article_num)
            if exact_results:
                return exact_results[:limit]

        if self._select_fusion_method(query) == "rrf":
            results = self._search_with_rrf(query, collection_name, limit * 2)
        else:
            results = self.weighted_fusion.search(query, collection_name, limit * 2)

        if collection_name == settings.default_collection:
            results = self.aggregate_by_article(results)
        return results[:limit]

    def aggregate_by_article(self, results: list[dict]) -> list[dict]:
        article_map = {}
        others = []

        for item in results:
            metadata = item.get("data", {})
            article_num = self._article_num_from_metadata(metadata)
            doc_type = metadata.get("type") or metadata.get("chunk_type") or ""

            if not article_num:
                match = re.search(r"第\s*(\d+)\s*条", item.get("text", ""))
                article_num = match.group(1) if match else ""
            if doc_type not in {"law_article", "parent"} or not article_num or article_num == "unknown":
                others.append(item)
                continue

            if article_num not in article_map:
                article_map[article_num] = item
                continue

            existing = article_map[article_num]
            if item.get("text", "") and item.get("text", "") not in existing.get("text", ""):
                existing["text"] = f"{existing.get('text', '')}\n{item.get('text', '')}"
            existing["score"] = max(existing.get("score", 0), item.get("score", 0))

        output = list(article_map.values()) + others
        return sorted(output, key=lambda item: item.get("score", 0), reverse=True)

    def search_article_exact(self, law_name: str, article_num: str) -> list[dict]:
        query = f"{law_name} 第{article_num}条" if law_name else f"第{article_num}条"

        for metadata_filter in self._article_filters(law_name, article_num):
            results = self.chroma_store.search(
                settings.default_collection,
                query,
                top_k=5,
                metadata_filter=metadata_filter,
            )
            if results:
                return results

        return self._text_filter_article(query, article_num)

    def get_stats(self) -> dict:
        return {
            settings.default_collection: self.chroma_store.get_count(settings.default_collection),
            "bm25_cache": self.bm25_cache.get_stats(),
        }

    def _select_fusion_method(self, query: str) -> str:
        if settings.fusion_strategy != "smart":
            return settings.fusion_strategy
        if re.search(r"区别|不同|对比|第\d+条", query):
            return "rrf"
        return "weighted"

    @staticmethod
    def _article_num_from_metadata(metadata: dict) -> str:
        article_num = metadata.get("article_num") or metadata.get("article_id") or ""
        return f"{article_num}" if article_num else ""

    @staticmethod
    def _article_filters(law_name: str, article_num: str) -> list[dict]:
        filters = [
            {"article_num": article_num, "type": "law_article"},
            {"article_id": article_num, "chunk_type": "parent"},
            {"article_id": article_num},
        ]
        if not law_name:
            return filters

        scoped_filters = []
        for metadata_filter in filters:
            scoped_filters.append({**metadata_filter, "law_name": law_name})
            scoped_filters.append({**metadata_filter, "source": law_name})
        return scoped_filters + filters

    def _text_filter_article(self, query: str, article_num: str) -> list[dict]:
        fallback = self.chroma_store.search(settings.default_collection, query, top_k=10)
        filtered = []
        for item in fallback:
            if re.search(rf"第\s*{article_num}\s*条", item.get("text", "")):
                filtered.append(item)
            if len(filtered) >= settings.agent_fallback_top_k:
                break
        return filtered

    def _search_with_rrf(self, query: str, collection: str, top_k: int) -> list[dict]:
        vector_results = self.chroma_store.search(collection, query, top_k=settings.vector_recall)
        if not vector_results:
            return []

        all_docs = self.chroma_store.get_all_documents(collection)
        if not all_docs:
            return vector_results[:top_k]

        keyword_results = self._keyword_search(query, collection, all_docs)
        return self._rrf_fusion(vector_results, keyword_results)[:top_k]

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

    @staticmethod
    def _rrf_fusion(results_a: list[dict], results_b: list[dict], k: int = 60) -> list[dict]:
        scores = {}
        result_map = {}

        for rank, item in enumerate(results_a, 1):
            doc_id = item.get("id") or f"doc_{hash(item.get('text', ''))}"
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            result_map[doc_id] = item

        for rank, item in enumerate(results_b, 1):
            doc_id = item.get("id") or f"doc_{hash(item.get('text', ''))}"
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            result_map.setdefault(doc_id, item)

        return [result_map[doc_id] for doc_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
