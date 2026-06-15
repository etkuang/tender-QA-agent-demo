# coding: utf-8

from typing import Protocol


class VectorStore(Protocol):
    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scalar_filter: str = "",
    ) -> list[dict]: ...

    def search_keyword(
        self,
        collection: str,
        query: str,
        top_k: int,
        scalar_filter: str = "",
    ) -> list[dict]: ...

    def query(
        self,
        collection: str,
        scalar_filter: str,
        limit: int,
    ) -> list[dict]: ...

    def get_by_ids(self, collection: str, document_ids: list[str]) -> list[dict]: ...

    def upsert(self, collection: str, rows: list[dict]) -> int: ...

    def create_version(self, collection: str) -> None: ...

    def delete_by_version(self, collection: str, data_version: str) -> int: ...

    def switch_alias(self, alias: str, collection: str) -> None: ...
