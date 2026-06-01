from typing import List
from sentence_transformers import SentenceTransformer
from agent_layer.infrastructure.config.agent_settings import settings


class EmbeddingService:
    """Singleton embedding service."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        result = self.model.encode([query], show_progress_bar=False)
        return result[0].tolist()
