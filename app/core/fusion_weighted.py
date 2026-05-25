# app/core/fusion_weighted.py
# -*- coding: utf-8 -*-
"""自定义融合检索器 - 分数归一化 + 加权融合 + 招投标专属Boost"""

import re
import jieba
from typing import List, Dict, Tuple
from collections import Counter

from config import settings
from app.storage.chroma_store import ChromaStore
from app.core.embedding import EmbeddingService
from app.core.bm25_cache import BM25Cache


class HybridFusionV2:
    """
    进阶版混合检索融合器 - 配置化版本
    所有关键词、权重、惩罚规则都从 settings 读取
    BM25 索引使用全局单例缓存
    """

    def __init__(self, bm25_cache: BM25Cache = None):
        self.chroma_store = ChromaStore()
        self.embedding_service = EmbeddingService()

        # 使用传入的 BM25 缓存
        self.bm25_cache = bm25_cache if bm25_cache is not None else BM25Cache()

        # 从配置加载所有硬编码内容
        self.TENDER_KEYWORDS = settings.tender_keywords
        self.REGULATION_KEYWORDS = settings.regulation_keywords
        self.PENALTY_PATTERNS = settings.penalty_patterns
        self.SEMANTIC_HEAVY_PATTERNS = settings.semantic_heavy_patterns
        self.KEYWORD_HEAVY_PATTERNS = settings.keyword_heavy_patterns
        self.WEIGHT_KEYWORD_HEAVY = settings.weights_keyword_heavy
        self.WEIGHT_SEMANTIC_HEAVY = settings.weights_semantic_heavy
        self.WEIGHT_BALANCED = settings.weights_balanced

    def _chinese_tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return BM25Cache._chinese_tokenize(text)

    def _get_bm25(self, collection: str, texts: List[str]):
        """获取或创建BM25索引"""
        return self.bm25_cache.get_or_build(collection, texts)

    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        """Min-Max归一化到[0,1]"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _compute_boost_score(self, text: str, metadata: Dict = None) -> float:
        """计算招投标专属Boost加分"""
        boost = 0.0
        hit_categories = set()
        text_lower = text.lower()

        # 判断是否为整体文件
        is_whole_doc = False
        if metadata:
            article_num = metadata.get("article_num", "")
            if article_num == "full":
                is_whole_doc = True


        # 整体文件：只给基础加分
        if is_whole_doc:
            if metadata and metadata.get("type") == "law_article":
                boost += 0.05
                hit_categories.add("law_article")
            if "招标" in text or "投标" in text:
                boost += 0.03
            return min(boost, 0.1)

        # 正常法条/论述类文档的加分逻辑
        if metadata and metadata.get("type") == "law_article":
            boost += settings.boost_law_article
            hit_categories.add("law_article")

        if re.match(r'^第\d+条', text.strip()):
            boost += settings.boost_article_start

        high_matches = 0
        medium_matches = 0
        low_matches = 0

        for kw in self.TENDER_KEYWORDS.get("high", []):
            if kw in text or kw.lower() in text_lower:
                high_matches += 1
        for kw in self.TENDER_KEYWORDS.get("medium", []):
            if kw in text or kw.lower() in text_lower:
                medium_matches += 1
        for kw in self.TENDER_KEYWORDS.get("low", []):
            if kw in text or kw.lower() in text_lower:
                low_matches += 1

        if high_matches >= 1:
            boost += settings.boost_high_keyword
            hit_categories.add("high")
        if medium_matches >= 2:
            boost += settings.boost_medium_keyword
            hit_categories.add("medium")
        if low_matches >= 2:
            boost += settings.boost_low_keyword
            hit_categories.add("low")

        reg_high_matches = 0
        for kw in self.REGULATION_KEYWORDS.get("high", []):
            if kw in text or kw.lower() in text_lower:
                reg_high_matches += 1
        if reg_high_matches >= 1:
            boost += settings.boost_regulation_keyword
            hit_categories.add("regulation")

        if len(hit_categories) >= 2:
            boost += settings.boost_multi_category

        return min(boost, settings.boost_max)

    def _compute_penalty_score(self, text: str) -> float:
        """计算降权惩罚分数"""
        penalty = 0.0

        for pattern, penalty_val, reason in self.PENALTY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                penalty += penalty_val
                break

        return min(penalty, settings.penalty_min)

    def _detect_query_type(self, query: str) -> str:
        """检测问题类型，用于动态权重切换"""
        query_lower = query.lower()

        for pattern in self.KEYWORD_HEAVY_PATTERNS:
            if re.search(pattern, query_lower):
                return "keyword_heavy"

        for pattern in self.SEMANTIC_HEAVY_PATTERNS:
            if re.search(pattern, query_lower):
                return "semantic_heavy"

        return "balanced"

    def _get_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """根据问题类型动态调整BM25和Dense权重"""
        query_type = self._detect_query_type(query)

        if query_type == "keyword_heavy":
            return self.WEIGHT_KEYWORD_HEAVY
        elif query_type == "semantic_heavy":
            return self.WEIGHT_SEMANTIC_HEAVY
        else:
            return self.WEIGHT_BALANCED

    def search(self, query: str, collection: str, top_k: int = 5) -> List[Dict]:
        """进阶版混合检索"""
        print(f"\n🔍 [HybridFusionV2] 开始检索: {query[:50]}...")

        dense_raw = self.chroma_store.search(
            collection=collection,
            query=query,
            top_k=settings.vector_recall
        )
        print(f"  📊 Dense召回: {len(dense_raw)} 条")

        if not dense_raw:
            return []

        all_docs = self.chroma_store.get_all_documents(collection)
        if not all_docs:
            return dense_raw[:top_k]

        texts = [doc["text"] for doc in all_docs]
        bm25 = self._get_bm25(collection, texts)

        tokenized_query = self._chinese_tokenize(query)
        bm25_scores_raw = bm25.get_scores(tokenized_query)

        import numpy as np
        top_bm25_indices = np.argsort(bm25_scores_raw)[-settings.vector_recall:][::-1]

        bm25_raw = []
        for idx in top_bm25_indices:
            if bm25_scores_raw[idx] > 0:
                bm25_raw.append({
                    "id": all_docs[idx]["id"],
                    "score": float(bm25_scores_raw[idx]),
                    "data": all_docs[idx]["metadata"],
                    "text": all_docs[idx]["text"],
                    "raw_index": idx
                })
        print(f"  📊 BM25召回: {len(bm25_raw)} 条")

        dense_scores_raw = [r.get("score", 0) for r in dense_raw]
        bm25_scores_raw_list = [r["score"] for r in bm25_raw]

        dense_scores_norm = self._min_max_normalize(dense_scores_raw)
        bm25_scores_norm = self._min_max_normalize(bm25_scores_raw_list)

        for i, r in enumerate(dense_raw):
            r["norm_score"] = dense_scores_norm[i] if i < len(dense_scores_norm) else 0
        for i, r in enumerate(bm25_raw):
            r["norm_score"] = bm25_scores_norm[i] if i < len(bm25_scores_norm) else 0

        bm25_weight, dense_weight = self._get_dynamic_weights(query)
        print(f"  🎯 动态权重: BM25={bm25_weight}, Dense={dense_weight} (类型: {self._detect_query_type(query)})")

        all_results = {}
        query_type = self._detect_query_type(query)

        for r in dense_raw:
            doc_id = r.get("id") or hash(r.get("text", ""))
            base_score = dense_weight * r.get("norm_score", 0)

            is_law_article = r.get("data", {}).get("type") == "law_article"
            if is_law_article and query_type == "semantic_heavy":
                base_score = base_score * settings.semantic_law_article_penalty

            boost = self._compute_boost_score(r.get("text", ""), r.get("data", {}))
            penalty = self._compute_penalty_score(r.get("text", ""))
            final_score = base_score + boost + penalty

            all_results[doc_id] = {
                "id": doc_id,
                "text": r.get("text", ""),
                "data": r.get("data", {}),
                "score": r.get("score", 0),
                "norm_score": r.get("norm_score", 0),
                "base_score": base_score,
                "boost": boost,
                "penalty": penalty,
                "final_score": final_score,
                "source": "dense"
            }

        for r in bm25_raw:
            doc_id = r.get("id") or hash(r.get("text", ""))
            base_score = bm25_weight * r.get("norm_score", 0)

            is_law_article = r.get("data", {}).get("type") == "law_article"
            if is_law_article and query_type == "semantic_heavy":
                base_score = base_score * settings.semantic_law_article_penalty

            boost = self._compute_boost_score(r.get("text", ""), r.get("data", {}))
            penalty = self._compute_penalty_score(r.get("text", ""))
            final_score = base_score + boost + penalty

            if doc_id in all_results:
                existing = all_results[doc_id]
                if final_score > existing["final_score"]:
                    existing["final_score"] = final_score
                    existing["base_score"] = base_score
                    existing["source"] = "both"
                existing["boost"] = max(existing["boost"], boost)
                existing["penalty"] = min(existing["penalty"], penalty)
            else:
                all_results[doc_id] = {
                    "id": doc_id,
                    "text": r.get("text", ""),
                    "data": r.get("data", {}),
                    "score": r.get("score", 0),
                    "norm_score": r.get("norm_score", 0),
                    "base_score": base_score,
                    "boost": boost,
                    "penalty": penalty,
                    "final_score": final_score,
                    "source": "bm25"
                }

        filtered_results = []
        for doc_id, r in all_results.items():
            if r["final_score"] < settings.min_final_score:
                continue

            if r["source"] == "dense" and r["boost"] == 0 and r["norm_score"] > settings.high_norm_threshold:
                r["final_score"] -= settings.dense_no_boost_penalty
                r["penalty"] -= settings.dense_no_boost_penalty

            filtered_results.append(r)

        sorted_results = sorted(filtered_results, key=lambda x: x["final_score"], reverse=True)

        print(f"  ✅ 最终返回 Top-{top_k} 条 (共融合 {len(all_results)} 条)")
        for i, r in enumerate(sorted_results[:top_k]):
            print(f"    [{i + 1}] final={r['final_score']:.4f} | "
                  f"base={r['base_score']:.3f} | "
                  f"boost={r['boost']:.2f} | "
                  f"src={r['source']}")

        output = []
        for r in sorted_results[:top_k]:
            output.append({
                "id": r["id"],
                "text": r["text"],
                "data": r["data"],
                "score": r["final_score"],
                "original_score": r.get("score", 0),
                "boost": r["boost"],
                "final_score": r["final_score"]
            })

        return output

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "regulations": self.chroma_store.get_count("regulations"),
            "bm25_cache": self.bm25_cache.get_stats()
        }