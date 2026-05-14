"""混合检索器 - 向量检索 + BM25关键词检索 + RRF融合 + Reranker"""

import jieba
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.storage.chroma_store import ChromaStore
from app.core.embedding import EmbeddingService


def chinese_tokenize(text: str) -> List[str]:
    """中文分词 - 使用jieba"""
    return jieba.lcut(text)


class Reranker:
    """重排模型 - BGE-reranker-base"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_model(self):
        if self._model is None and settings.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder('BAAI/bge-reranker-base')
                print("Reranker model loaded")
            except Exception as e:
                print(f"Reranker load failed: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[int]:
        """
        重排序，返回排序后的索引

        参数:
            query: 用户问题
            documents: 文档文本列表
            top_k: 返回数量

        返回:
            排序后的索引列表
        """
        model = self._get_model()
        if model is None or not documents:
            return list(range(min(top_k, len(documents))))

        pairs = [[query, doc[:512]] for doc in documents]
        scores = model.predict(pairs)
        indices = np.argsort(scores)[::-1]
        return indices[:top_k].tolist()


class HybridRetriever:
    """
    混合检索器

    检索策略：
    1. 向量检索（Chroma）- 召回50条
    2. 关键词检索（BM25 + jieba分词）- 召回50条
    3. RRF融合 - 合并去重
    4. Reranker精排 - 重新排序
    5. 返回Top-K
    """

    def __init__(self):
        self.chroma_store = ChromaStore()
        self.embedding_service = EmbeddingService()
        self.reranker = Reranker()

        # BM25索引缓存
        self._bm25_indices = {}
        self._bm25_texts = {}

    def _get_bm25(self, collection: str, texts: List[str]):
        """获取或创建BM25索引（使用jieba分词）"""
        if collection not in self._bm25_indices:
            tokenized = [chinese_tokenize(text) for text in texts]
            self._bm25_indices[collection] = BM25Okapi(tokenized)
            self._bm25_texts[collection] = texts
        return self._bm25_indices[collection]

    def search(self, query: str, collection: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索

        参数:
            query: 用户问题
            collection: 集合名称（bids/regulations/prices）
            top_k: 返回数量

        返回:
            检索结果列表
        """

        # 1. 向量检索（召回50条）
        vector_results = self.chroma_store.search(
            collection=collection,
            query=query,
            top_k=settings.vector_recall
        )
        if not vector_results:
            return []

        # 2. 获取所有文档用于BM25
        all_docs = self.chroma_store.get_all_documents(collection)
        if not all_docs:
            return vector_results[:top_k]

        texts = [doc["text"] for doc in all_docs]
        bm25 = self._get_bm25(collection, texts)

        # 3. BM25关键词检索
        tokenized_query = chinese_tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
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

        # 4. RRF融合
        fused = self._rrf_fusion(vector_results, keyword_results, k=60)

        # 5. Reranker精排
        if settings.use_reranker and len(fused) > 0:
            documents = [f["text"][:500] for f in fused[:30]]
            rerank_indices = self.reranker.rerank(query, documents, top_k)
            fused = [fused[i] for i in rerank_indices if i < len(fused)]

        # 6. 返回Top-K
        return fused[:top_k]

    def _rrf_fusion(self, results_a: List, results_b: List, k: int = 60) -> List:
        """RRF融合算法"""
        scores = {}
        result_map = {}

        for rank, r in enumerate(results_a, 1):
            doc_id = r.get("id", r.get("data", {}).get("id", rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            result_map[doc_id] = r

        for rank, r in enumerate(results_b, 1):
            doc_id = r.get("id", r.get("data", {}).get("id", rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in result_map:
                result_map[doc_id] = r

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [result_map[doc_id] for doc_id, _ in sorted_ids if doc_id in result_map]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "bids": self.chroma_store.get_count("bids"),
            "regulations": self.chroma_store.get_count("regulations"),
            "prices": self.chroma_store.get_count("prices"),
        }
