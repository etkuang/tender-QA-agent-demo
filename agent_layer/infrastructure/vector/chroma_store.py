import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from agent_layer.infrastructure.config.agent_settings import settings


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
            persist_dir = settings.chroma_persist_dir_path
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir.as_posix())
        return self._client

    def _get_embedding_function(self):
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

    def get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            embedding_function = self._get_embedding_function()
            try:
                collection = client.get_collection(name=name, embedding_function=embedding_function)
            except Exception:
                collection = client.create_collection(
                    name=name,
                    embedding_function=embedding_function
                )
            self._collections[name] = collection
        return self._collections[name]

    def add_documents(self, collection: str, texts: List[str], metadatas: List[Dict], ids: List[str]):
        if not texts:
            return
        collection_obj = self.get_collection(collection)
        batch_size = 100
        total = len(texts)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            collection_obj.add(
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            print(f"[Chroma] add batch: {end}/{total} ({end / total:.1%})")
        print(f"Added {len(texts)} documents to '{collection}'")

    def search(self, collection: str, query: str, top_k: int = 10, metadata_filter: Optional[Dict] = None) -> List[
        Dict]:
        collection_obj = self.get_collection(collection)

        # Convert multi-field metadata filter to Chroma's $and syntax.
        where_filter = None
        if metadata_filter:
            if len(metadata_filter) == 1:
                key, value = list(metadata_filter.items())[0]
                where_filter = {key: value}
            else:
                conditions = [{k: v} for k, v in metadata_filter.items()]
                where_filter = {"$and": conditions}

        results = collection_obj.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        documents = []
        if results and results.get('documents') and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                doc_id = results['ids'][0][i] if results.get('ids') else str(i)
                distance = results['distances'][0][i] if results.get('distances') else 0
                score = 1 / (1 + distance)
                documents.append({
                    "id": doc_id,
                    "text": doc,
                    "data": metadata,
                    "score": score
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
        except:
            pass