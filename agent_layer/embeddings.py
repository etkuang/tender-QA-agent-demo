# coding: utf-8
# @Author: Wang Qingkang

from threading import Semaphore

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from agent_layer.config import settings


class EvidenceEmbeddings(Embeddings):
    """One shared, pinned encoder for evidence documents and questions."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model
        self._semaphore = Semaphore(settings.embedding_concurrency_limit)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "")

    def embed_query(self, text: str) -> list[float]:
        return self._encode(
            [text],
            "Instruct: Given a question, retrieve relevant passages "
            "that provide evidence for answering it.\nQuery:",
        )[0]

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        # LangChain's async embedding methods offload these synchronous methods.
        with self._semaphore:
            return self._model.encode(
                texts,
                prompt=prompt,
                batch_size=1,
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            ).tolist()


def build_embedding_model() -> Embeddings:
    """Load once during application composition, not inside graph nodes."""
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        device="cpu",
        model_kwargs={"dtype": "float32"},
        trust_remote_code=False,
    )
    return EvidenceEmbeddings(model)