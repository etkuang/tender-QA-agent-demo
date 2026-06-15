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
        positions = {}
        for chunk in chunks:
            metadata = chunk.get("data", {})
            parent = parent_map.get(metadata.get("parent_id"))
            if parent:
                item = {
                    **parent,
                    "score": chunk.get("score"),
                    "vector_rank": chunk.get("vector_rank"),
                    "bm25_rank": chunk.get("bm25_rank"),
                    "fusion_score": chunk.get("fusion_score"),
                }
            else:
                item = chunk
            document_id = item.get("id")
            if document_id in positions:
                position = positions[document_id]
                current_score = output[position].get("score")
                candidate_score = item.get("score")
                if candidate_score is not None and (current_score is None or candidate_score > current_score):
                    output[position] = item
                continue
            if document_id:
                positions[document_id] = len(output)
            output.append(item)
        return output
