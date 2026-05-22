# app/core/bm25_cache.py
"""
BM25 索引缓存 - 单例模式
确保整个应用只有一份 BM25 索引，避免内存浪费
"""
import jieba
from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional


class BM25Cache:
    """BM25 索引全局缓存（单例）"""
    _instance = None
    _bm25_indices: Dict[str, BM25Okapi] = {}
    _bm25_texts: Dict[str, List[str]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _chinese_tokenize(text: str) -> List[str]:
        """中文分词（静态方法，可被外部调用）"""
        if not text:
            return []
        return [word for word in jieba.cut(str(text)) if word.strip()]

    def get_or_build(self, collection: str, texts: List[str]) -> BM25Okapi:
        """
        获取或构建 BM25 索引
        相同 collection 只构建一次
        """
        if collection not in self._bm25_indices:
            if not texts:
                raise ValueError(f"无法为 collection '{collection}' 构建 BM25：texts 为空")

            print(f"🔄 构建 BM25 索引: {collection}，共 {len(texts)} 条文档")
            tokenized = [self._chinese_tokenize(text) for text in texts]
            self._bm25_indices[collection] = BM25Okapi(tokenized)
            self._bm25_texts[collection] = texts
            print(f"✅ BM25 索引构建完成: {collection}")

        return self._bm25_indices[collection]

    def get_texts(self, collection: str) -> List[str]:
        """获取 collection 对应的原始文本列表"""
        return self._bm25_texts.get(collection, [])

    def has_index(self, collection: str) -> bool:
        """检查某个 collection 的 BM25 索引是否已存在"""
        return collection in self._bm25_indices

    def invalidate(self, collection: str):
        """
        使指定 collection 的 BM25 缓存失效
        知识库更新后调用此方法，下次检索时会重建索引
        """
        if collection in self._bm25_indices:
            self._bm25_indices.pop(collection)
            self._bm25_texts.pop(collection)
            print(f"🗑️ BM25 缓存已失效: {collection}")

    def invalidate_all(self):
        """使所有 BM25 缓存失效"""
        count = len(self._bm25_indices)
        self._bm25_indices.clear()
        self._bm25_texts.clear()
        print(f"🗑️ BM25 缓存全部失效，共 {count} 个 collection")

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "cached_collections": list(self._bm25_indices.keys()),
            "total_collections": len(self._bm25_indices),
            "total_documents": sum(len(texts) for texts in self._bm25_texts.values())
        }