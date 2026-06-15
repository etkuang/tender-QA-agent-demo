# coding: utf-8

from knowledge_base_layer.retrieval.store import VectorStore


class ParentContextExpander:
    def __init__(self, store: VectorStore):
        self.store = store

    def expand(self, collection: str, chunks: list[dict]) -> list[dict]:
        parent_ids = []
        for chunk in chunks:
            metadata = chunk.get("data", {})
            if metadata.get("chunk_type") == "child" and metadata.get("parent_id"):
                parent_ids.append(metadata["parent_id"])
        parents = self.store.get_by_ids(collection, list(dict.fromkeys(parent_ids)))
        parent_map = {parent["id"]: parent for parent in parents}
        output = []
        for chunk in chunks:
            metadata = chunk.get("data", {})
            parent = parent_map.get(metadata.get("parent_id"))
            if parent:
                parent["score"] = chunk.get("score")
                parent["vector_rank"] = chunk.get("vector_rank")
                parent["bm25_rank"] = chunk.get("bm25_rank")
                parent["fusion_score"] = chunk.get("fusion_score")
                output.append(parent)
            else:
                output.append(chunk)
        return output
