# app/core/retriever.py
from config import settings

# 导入新融合器
from app.core.fusion_weighted import HybridFusionV2
import jieba
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from app.storage.chroma_store import ChromaStore
from app.core.embedding import EmbeddingService
from app.core.bm25_cache import BM25Cache
from app.utils.chinese_number import chinese_number_converter  # ← 新增
import re


class HybridRetriever:
    def __init__(self):
        self.chroma_store = ChromaStore()
        self.embedding_service = EmbeddingService()

        # 使用单例 BM25 缓存
        self.bm25_cache = BM25Cache()

        # 根据配置选择融合器
        fusion_strategy = settings.fusion_strategy
        if fusion_strategy == "weighted":
            self.weighted_fusion = HybridFusionV2(bm25_cache=self.bm25_cache)
            print("✅ 使用 Weighted Fusion (分数归一化+加权+Boost)")
        elif fusion_strategy == "rrf":
            print("✅ 使用 RRF Fusion (原方案)")
        else:  # "smart" 或其他，默认使用 weighted
            self.weighted_fusion = HybridFusionV2(bm25_cache=self.bm25_cache)
            print("✅ 使用 Weighted Fusion (智能模式)")

    def aggregate_by_article(self, results: List[Dict]) -> List[Dict]:
        """按法条编号聚合检索结果"""
        if not results:
            return results

        article_map = {}
        other_results = []

        for r in results:
            data = r.get("data", {})
            if not data:
                data = r.get("metadata", {})

            doc_type = data.get("type", "")
            article_num = data.get("article_num", "")

            if (doc_type == "law_article" or "law" in r.get("text", "").lower()) and (
                    not article_num or article_num == "unknown"):
                text = r.get("text", "")
                match = re.search(r'第(\d+)条', text)
                if match:
                    article_num = match.group(1)
                    data["article_num"] = article_num

            if doc_type != "law_article" or not article_num or article_num == "unknown":
                other_results.append(r)
                continue

            if article_num not in article_map:
                article_map[article_num] = {
                    "id": r.get("id", ""),
                    "text": r.get("text", ""),
                    "data": data,
                    "score": r.get("score", 0)
                }
            else:
                existing = article_map[article_num]
                existing_text = existing["text"]
                new_text = r.get("text", "")
                if new_text and new_text not in existing_text:
                    existing["text"] = existing_text + "\n" + new_text
                existing["score"] = max(existing["score"], r.get("score", 0))

        aggregated = list(article_map.values()) + other_results
        aggregated.sort(key=lambda x: x["score"], reverse=True)

        print(f"  📊 按法条聚合: {len(results)} 条 → {len(aggregated)} 条")
        for art in aggregated[:3]:
            art_num = art.get("data", {}).get("article_num", "?")
            if art_num != "?":
                print(f"    聚合到第{art_num}条")

        return aggregated

    def search(self, query: str, collection: str, top_k: int = 5) -> List[Dict]:
        """智能路由 + 精确法条预处理"""
        import re

        # 精确法条查询预处理
        article_match = re.search(r'第(\d+)条', query)
        if article_match and len(query) < 30:
            article_num = article_match.group(1)
            print(f"  📖 精确法条查询: 第{article_num}条")

            exact_results = self.chroma_store.search(
                collection=collection,
                query=f"第{article_num}条",
                top_k=top_k,
                metadata_filter={"type": "law_article", "article_num": article_num}
            )

            if exact_results:
                print(f"  ✅ 精确匹配成功")
                return exact_results

            all_results = self.chroma_store.search(
                collection=collection,
                query=f"第{article_num}条",
                top_k=10
            )
            filtered = [r for r in all_results if re.search(rf'第\s*{article_num}\s*条', r.get("text", ""))]
            if filtered:
                return filtered[:top_k]

        # 智能选择融合器
        fusion_method = self._select_fusion_method(query)
        print(f"  🎯 智能选择: {fusion_method}")

        recall_k = top_k * 2

        if fusion_method == "weighted":
            results = self.weighted_fusion.search(query, collection, recall_k)
        else:
            results = self._search_with_rrf(query, collection, recall_k)

        if collection == "regulations":
            results = self.aggregate_by_article(results)

        return results[:top_k]

    def _select_fusion_method(self, query: str) -> str:
        """根据问题类型选择融合器"""
        if hasattr(settings, 'fusion_strategy') and settings.fusion_strategy != "smart":
            print(f"  🎯 使用配置指定策略: {settings.fusion_strategy}")
            return settings.fusion_strategy

        import re
        if re.search(r'什么是|定义|解释|含义|如何|怎么|步骤|流程|多少|交|办', query):
            return "weighted"
        if re.search(r'区别|不同|对比|第\d+条', query):
            return "rrf"
        if re.search(r'处罚|罚款|责任', query):
            return "weighted"
        if len(query) > 15:
            return "weighted"
        return "weighted"

    def _search_with_rrf(self, query: str, collection: str, top_k: int = 5) -> List[Dict]:
        """RRF 融合逻辑"""
        vector_results = self.chroma_store.search(
            collection=collection,
            query=query,
            top_k=settings.vector_recall
        )

        if not vector_results:
            return []

        all_docs = self.chroma_store.get_all_documents(collection)
        if not all_docs:
            return vector_results[:top_k]

        texts = [doc["text"] for doc in all_docs]
        bm25 = self._get_bm25(collection, texts)

        tokenized_query = self._chinese_tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        import numpy as np
        top_bm25_indices = np.argsort(bm25_scores)[-settings.vector_recall:][::-1]

        keyword_results = []
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0:
                keyword_results.append({
                    "id": all_docs[idx]["id"],
                    "score": float(bm25_scores[idx]),
                    "data": all_docs[idx]["metadata"],
                    "text": all_docs[idx]["text"]
                })

        fused = self._rrf_fusion(vector_results, keyword_results, k=60)
        return fused[:top_k]

    def _rrf_fusion(self, results_a: List, results_b: List, k: int = 60) -> List:
        """RRF融合算法"""
        scores = {}
        result_map = {}

        for rank, r in enumerate(results_a, 1):
            doc_id = r.get("id") or r.get("data", {}).get("id") or str(hash(r.get("text", "")))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            result_map[doc_id] = r

        for rank, r in enumerate(results_b, 1):
            doc_id = r.get("id") or r.get("data", {}).get("id") or str(hash(r.get("text", "")))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in result_map:
                result_map[doc_id] = r

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [result_map[doc_id] for doc_id, _ in sorted_ids if doc_id in result_map]

    def _chinese_tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return BM25Cache._chinese_tokenize(text)

    def _get_bm25(self, collection: str, texts: List[str]):
        """获取或创建BM25索引"""
        return self.bm25_cache.get_or_build(collection, texts)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "bids": self.chroma_store.get_count("bids"),
            "regulations": self.chroma_store.get_count("regulations"),
            "prices": self.chroma_store.get_count("prices"),
            "bm25_cache": self.bm25_cache.get_stats()
        }

    def search_article_exact(self, law_name: str, article_num: str) -> List[Dict]:
        """精确检索指定法条"""
        results = self.chroma_store.search(
            collection="regulations",
            query=f"{law_name} 第{article_num}条",
            top_k=5,
            metadata_filter={"article_num": article_num, "type": "law_article"}
        )

        if results:
            return results

        results = self.search(f"第{article_num}条", "regulations", top_k=10)
        filtered = []

        # 使用配置化的中文数字转换
        for r in results:
            text = r.get("text", "")
            if re.search(rf'第\s*{article_num}\s*条', text):
                filtered.append(r)
            else:
                # 尝试中文数字匹配（使用配置化的转换器）
                chinese_num = chinese_number_converter.to_arabic(article_num)
                if chinese_num != article_num and re.search(rf'第\s*{chinese_num}\s*条', text):
                    filtered.append(r)
            if len(filtered) >= 3:
                break

        return filtered

    def search_with_filter(self, query: str, collection: str, metadata_filter: Dict, top_k: int = 5) -> List[Dict]:
        """带元数据过滤的检索"""
        return self.chroma_store.search(
            collection=collection,
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter
        )