# coding: utf-8

from typing import Protocol

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from agent_layer.config import Settings


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


class ModelFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.effective_llm_api_key or "missing-key",
            base_url=self.settings.llm_api_url,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
            timeout=self.settings.llm_timeout_seconds,
        )

    def build_embedding_gateway(self) -> EmbeddingGateway:
        return LocalEmbeddingGateway(self.settings.embedding_model_name_or_path)
