# coding: utf-8

import json

import httpx

from knowledge_base_layer.config import Settings


class MilvusStore:
    """Milvus REST v2 adapter used by online retrieval and release switching."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        if client is not None:
            self.client = client
            return
        headers = {"Content-Type": "application/json"}
        if settings.milvus_token:
            headers["Authorization"] = f"Bearer {settings.milvus_token}"
        self.client = httpx.Client(
            base_url=settings.milvus_uri.rstrip("/"),
            headers=headers,
            timeout=settings.milvus_timeout_seconds,
        )

    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scalar_filter: str = "",
    ) -> list[dict]:
        payload = self._base_payload(collection)
        payload.update(
            {
                "data": [query_vector],
                "annsField": self.settings.milvus_dense_field,
                "limit": top_k,
                "outputFields": self.settings.milvus_output_fields,
                "searchParams": {"metric_type": self.settings.milvus_metric_type},
            }
        )
        if scalar_filter:
            payload["filter"] = scalar_filter
        response = self._post("/v2/vectordb/entities/search", payload)
        return self._search_documents(response)

    def search_keyword(
        self,
        collection: str,
        query: str,
        top_k: int,
        scalar_filter: str = "",
    ) -> list[dict]:
        payload = self._base_payload(collection)
        payload.update(
            {
                "data": [query],
                "annsField": self.settings.milvus_sparse_field,
                "limit": top_k,
                "outputFields": self.settings.milvus_output_fields,
                "searchParams": {"metric_type": "BM25", "params": {}},
            }
        )
        if scalar_filter:
            payload["filter"] = scalar_filter
        response = self._post("/v2/vectordb/entities/search", payload)
        return self._search_documents(response)

    def query(self, collection: str, scalar_filter: str, limit: int) -> list[dict]:
        payload = self._base_payload(collection)
        payload.update(
            {
                "filter": scalar_filter,
                "limit": limit,
                "outputFields": self.settings.milvus_output_fields,
            }
        )
        response = self._post("/v2/vectordb/entities/query", payload)
        return [self._entity_to_document(entity) for entity in response]

    def get_by_ids(self, collection: str, document_ids: list[str]) -> list[dict]:
        if not document_ids:
            return []
        payload = self._base_payload(collection)
        payload.update(
            {
                "id": document_ids,
                "outputFields": self.settings.milvus_output_fields,
            }
        )
        response = self._post("/v2/vectordb/entities/get", payload)
        return [self._entity_to_document(entity) for entity in response]

    def upsert(self, collection: str, rows: list[dict]) -> int:
        payload = self._base_payload(collection)
        payload["data"] = rows
        response = self._post("/v2/vectordb/entities/upsert", payload)
        return response.get("upsertCount", len(rows)) if isinstance(response, dict) else len(rows)

    def create_version(self, collection: str) -> None:
        schema = {
            "autoId": False,
            "enableDynamicField": False,
            "fields": [
                self._varchar_field("id", 512, is_primary=True),
                self._varchar_field("content", 65535, enable_analyzer=True),
                self._varchar_field("title", 2048),
                self._varchar_field("source_url", 4096),
                self._varchar_field("document_id", 512),
                self._varchar_field("law_name", 1024),
                self._varchar_field("article_id", 128),
                self._varchar_field("entity_id", 512),
                self._varchar_field("published_at", 64),
                {"fieldName": "authority_level", "dataType": "Int64"},
                {"fieldName": "freshness_level", "dataType": "Int64"},
                self._varchar_field("parent_id", 512),
                self._varchar_field("chunk_type", 64),
                self._varchar_field("validity_status", 64),
                self._varchar_field("region", 128),
                self._varchar_field("effective_date", 64),
                self._varchar_field("end_date", 64),
                self._varchar_field("source_kind", 64),
                self._varchar_field("source_as_of", 64),
                self._varchar_field("data_version", 128),
                self._varchar_field("metadata", 65535),
                {
                    "fieldName": self.settings.milvus_dense_field,
                    "dataType": "FloatVector",
                    "elementTypeParams": {"dim": f"{self.settings.embedding_dimension}"},
                },
                {
                    "fieldName": self.settings.milvus_sparse_field,
                    "dataType": "SparseFloatVector",
                },
            ],
            "functions": [
                {
                    "name": "content_bm25",
                    "type": "BM25",
                    "inputFieldNames": [self.settings.milvus_content_field],
                    "outputFieldNames": [self.settings.milvus_sparse_field],
                    "params": {},
                }
            ],
        }
        index_params = [
            {
                "fieldName": self.settings.milvus_dense_field,
                "metricType": self.settings.milvus_metric_type,
                "indexName": self.settings.milvus_dense_field,
                "indexType": "HNSW",
                "params": {
                    "M": self.settings.milvus_hnsw_m,
                    "efConstruction": self.settings.milvus_hnsw_ef_construction,
                },
            },
            {
                "fieldName": self.settings.milvus_sparse_field,
                "metricType": "BM25",
                "indexName": self.settings.milvus_sparse_field,
                "indexType": "SPARSE_INVERTED_INDEX",
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": self.settings.milvus_bm25_k1,
                    "bm25_b": self.settings.milvus_bm25_b,
                },
            },
        ]
        payload = self._base_payload(collection)
        payload.update({"schema": schema, "indexParams": index_params})
        self._post("/v2/vectordb/collections/create", payload)

    def delete_by_version(self, collection: str, data_version: str) -> int:
        payload = self._base_payload(collection)
        payload["filter"] = f'data_version == "{self._escape(data_version)}"'
        response = self._post("/v2/vectordb/entities/delete", payload)
        return response.get("deleteCount", 0) if isinstance(response, dict) else 0

    def switch_alias(self, alias: str, collection: str) -> None:
        describe_payload = {"aliasName": alias, "dbName": self.settings.milvus_database}
        try:
            self._post("/v2/vectordb/aliases/describe", describe_payload)
        except RuntimeError:
            create_payload = {
                "aliasName": alias,
                "collectionName": collection,
                "dbName": self.settings.milvus_database,
            }
            self._post("/v2/vectordb/aliases/create", create_payload)
            return
        alter_payload = {
            "aliasName": alias,
            "collectionName": collection,
            "dbName": self.settings.milvus_database,
        }
        self._post("/v2/vectordb/aliases/alter", alter_payload)

    def _post(self, endpoint: str, payload: dict):
        response = self.client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Milvus request failed: {body.get('message') or body}")
        return body.get("data", {})

    def _base_payload(self, collection: str) -> dict:
        return {
            "collectionName": collection,
            "dbName": self.settings.milvus_database,
        }

    def _search_documents(self, response) -> list[dict]:
        if not response:
            return []
        hits = response[0] if isinstance(response, list) and response and isinstance(response[0], list) else response
        return [self._entity_to_document(hit, hit.get("distance")) for hit in hits]

    def _entity_to_document(self, entity: dict, score: float | None = None) -> dict:
        metadata = dict(entity)
        document_id = metadata.pop(self.settings.milvus_primary_field, metadata.pop("id", ""))
        content = metadata.pop(self.settings.milvus_content_field, "")
        metadata.pop("distance", None)
        raw_metadata = metadata.pop("metadata", None)
        if isinstance(raw_metadata, str):
            try:
                raw_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                raw_metadata = {"raw_metadata": raw_metadata}
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        return {
            "id": f"{document_id}",
            "text": content,
            "data": metadata,
            "score": score,
        }

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _varchar_field(
        name: str,
        max_length: int,
        is_primary: bool = False,
        enable_analyzer: bool = False,
    ) -> dict:
        field = {
            "fieldName": name,
            "dataType": "VarChar",
            "elementTypeParams": {"max_length": max_length},
        }
        if is_primary:
            field["isPrimary"] = True
        if enable_analyzer:
            field["elementTypeParams"]["enable_analyzer"] = True
            field["elementTypeParams"]["enable_match"] = True
        return field
