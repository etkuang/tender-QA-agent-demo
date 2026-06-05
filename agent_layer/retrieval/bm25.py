# coding: utf-8

import jieba
from rank_bm25 import BM25Okapi


class BM25Cache:
    _instance = None
    _indices = {}
    _texts = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not text:
            return []
        return [word for word in jieba.cut(text) if word.strip()]

    def get_or_build(self, collection: str, texts: list[str]) -> BM25Okapi:
        if collection not in self._indices:
            tokenized = [self.tokenize(text) for text in texts]
            self._indices[collection] = BM25Okapi(tokenized)
            self._texts[collection] = texts
        return self._indices[collection]

    def get_stats(self) -> dict:
        return {
            "cached_collections": list(self._indices.keys()),
            "total_collections": len(self._indices),
            "total_documents": sum(len(texts) for texts in self._texts.values()),
        }
