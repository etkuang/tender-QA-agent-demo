# coding: utf-8

import chromadb
from chromadb.utils import embedding_functions

from common.logging import get_logger
from agent_layer.config import settings

logger = get_logger("agent.retrieval.chroma")


class ChromaStore:
    _instance = None
    _client = None
    _collections = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self):
        if self._client is None:
            settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_path.as_posix())
        return self._client

    @staticmethod
    def _get_embedding_function():
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name_or_path,
        )

    def get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            embedding_function = self._get_embedding_function()
            try:
                collection = client.get_collection(name=name, embedding_function=embedding_function)
            except Exception:
                logger.warning("chroma collection unavailable; creating collection | name=%s", name, exc_info=True)
                collection = client.create_collection(name=name, embedding_function=embedding_function)
            self._collections[name] = collection
        return self._collections[name]

    @staticmethod
    def _build_where_filter(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            key, value = next(iter(metadata_filter.items()))
            return {key: value}
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        collection_obj = self.get_collection(collection)
        results = collection_obj.query(
            query_texts=[query],
            n_results=top_k,
            where=self._build_where_filter(metadata_filter),
        )

        documents = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return documents

        for index, text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][index] if results.get("metadatas") else {}
            doc_id = results["ids"][0][index] if results.get("ids") else f"doc_{index}"
            distance = results["distances"][0][index] if results.get("distances") else 0
            documents.append(
                {
                    "id": doc_id,
                    "text": text,
                    "data": metadata,
                    "score": 1 / (1 + distance),
                }
            )
        return documents

    def get_all_documents(self, collection: str) -> list[dict]:
        collection_obj = self.get_collection(collection)
        results = collection_obj.get()
        documents = []
        if not results or not results.get("ids"):
            return documents

        for index, doc_id in enumerate(results["ids"]):
            documents.append(
                {
                    "id": doc_id,
                    "text": results["documents"][index] if results.get("documents") else "",
                    "metadata": results["metadatas"][index] if results.get("metadatas") else {},
                }
            )
        return documents

    def get_count(self, collection: str) -> int:
        return self.get_collection(collection).count()
