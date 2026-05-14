"""Chroma向量数据库客户端"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Callable
import pandas as pd
from pathlib import Path

from app.config import settings


class ChromaStore:
    """Chroma向量数据库客户端（单例模式）"""

    _instance = None
    _client = None
    _collections = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            print(f"Chroma client initialized, persist dir: {settings.chroma_persist_dir}")
        return self._client

    def _get_embedding_function(self):
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

    def get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            try:
                collection = client.get_collection(name)
            except:
                collection = client.create_collection(
                    name=name, embedding_function=self._get_embedding_function()
                )
            self._collections[name] = collection
        return self._collections[name]

    def add_documents(self, collection: str, texts: List[str], metadatas: List[Dict], ids: List[str]):
        if not texts:
            return
        collection_obj = self.get_collection(collection)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            collection_obj.add(
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
        print(f"Added {len(texts)} documents to '{collection}'")

    # def search(self, collection: str, query: str, top_k: int = 10) -> List[Dict]:
    #     collection_obj = self.get_collection(collection)
    #     results = collection_obj.query(query_texts=[query], n_results=top_k)
    #     documents = []
    #     if results and results.get('documents') and results['documents'][0]:
    #         for i, doc in enumerate(results['documents'][0]):
    #             metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
    #             doc_id = results['ids'][0][i] if results.get('ids') else str(i)
    #             distance = results['distances'][0][i] if results.get('distances') else 0
    #             score = 1 / (1 + distance)
    #             documents.append({
    #                 "id": doc_id, "text": doc, "metadata": metadata,
    #                 "score": score, "data": metadata
    #             })
    #     return documents
    def search(self, collection: str, query: str, top_k: int = 10) -> List[Dict]:
        collection_obj = self.get_collection(collection)

        # 手动生成查询向量，确保使用正确的模型
        from app.core.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.embed_query(query)

        # 使用向量查询
        results = collection_obj.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = []
        if results and results.get('documents') and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                doc_id = results['ids'][0][i] if results.get('ids') else str(i)
                distance = results['distances'][0][i] if results.get('distances') else 0
                score = 1 / (1 + distance)
                documents.append({
                    "id": doc_id, "text": doc, "metadata": metadata,
                    "score": score, "data": metadata
                })
        return documents

    def get_all_documents(self, collection: str) -> List[Dict]:
        collection_obj = self.get_collection(collection)
        try:
            results = collection_obj.get()
            documents = []
            if results and results.get('ids'):
                for i, doc_id in enumerate(results['ids']):
                    doc = {
                        "id": doc_id,
                        "text": results['documents'][i] if results.get('documents') else "",
                        "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                    }
                    documents.append(doc)
            return documents
        except Exception as e:
            print(f"Get all documents error: {e}")
            return []

    def get_count(self, collection: str) -> int:
        try:
            return self.get_collection(collection).count()
        except:
            return 0

    def delete_collection(self, collection: str):
        if collection in self._collections:
            del self._collections[collection]
        try:
            self._get_client().delete_collection(collection)
            print(f"Deleted collection '{collection}'")
        except:
            pass

    def rebuild_from_excel(self, collection: str, excel_path: str, text_builder_func: Callable):
        if not Path(excel_path).exists():
            print(f"File not found: {excel_path}")
            return
        df = pd.read_excel(excel_path)
        data = df.to_dict('records')
        if not data:
            return
        texts, metadatas, ids = [], [], []
        for i, record in enumerate(data):
            text = text_builder_func(record)
            if text and len(text) > 10:
                texts.append(text)
                metadatas.append(record)
                ids.append(f"{collection}_{i}_{hash(text) % 10000}")
        self.delete_collection(collection)
        self.add_documents(collection, texts, metadatas, ids)
        print(f"Rebuilt '{collection}' with {len(texts)} documents")