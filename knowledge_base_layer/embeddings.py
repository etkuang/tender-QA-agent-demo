# coding: utf-8
# @Author: Wang Qingkang

from typing import Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingGateway(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingGateway:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model = None

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name_or_path)
        return self.model