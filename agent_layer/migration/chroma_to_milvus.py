# coding: utf-8

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from pydantic import BaseModel, Field

from agent_layer.ingestion.normalizer import ChunkNormalizer, NormalizationIssue
from agent_layer.models import EmbeddingGateway
from agent_layer.retrieval.milvus import MilvusStore


class MigrationManifest(BaseModel):
    ingestion_run_id: str
    source_path: str
    source_collection: str
    target_collection: str
    target_alias: str
    data_version: str
    source_hash: str
    source_count: int
    normalized_count: int
    written_count: int
    issues: list[NormalizationIssue] = Field(default_factory=list)
    published: bool = False


class ChromaToMilvusMigrator:
    def __init__(
        self,
        chroma_path: Path,
        milvus: MilvusStore,
        embeddings: EmbeddingGateway,
        normalizer: ChunkNormalizer | None = None,
    ):
        self.chroma_path = chroma_path
        self.milvus = milvus
        self.embeddings = embeddings
        self.normalizer = normalizer or ChunkNormalizer()

    def migrate(
        self,
        source_collection: str,
        target_alias: str,
        data_version: str,
        batch_size: int = 128,
        publish: bool = False,
    ) -> MigrationManifest:
        ingestion_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_collection = f"{target_alias}_{data_version}"
        source_hash = self._source_hash(self.chroma_path / "chroma.sqlite3")
        records = self._read_source(source_collection)
        chunks = []
        issues = []
        for record in records:
            chunk, record_issues = self.normalizer.normalize(
                record["id"],
                record["content"],
                record["metadata"],
                data_version,
            )
            chunks.append(chunk)
            issues.extend(record_issues)

        critical_issues = [issue for issue in issues if issue.critical]
        if critical_issues:
            return MigrationManifest(
                ingestion_run_id=ingestion_run_id,
                source_path=self.chroma_path.as_posix(),
                source_collection=source_collection,
                target_collection=target_collection,
                target_alias=target_alias,
                data_version=data_version,
                source_hash=source_hash,
                source_count=len(records),
                normalized_count=len(chunks),
                written_count=0,
                issues=issues,
                published=False,
            )

        self.milvus.create_version(target_collection)
        written_count = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_documents([chunk.content for chunk in batch])
            rows = []
            for chunk, vector in zip(batch, vectors, strict=True):
                rows.append(chunk.model_copy(update={"dense_vector": vector}).model_dump())
            written_count += self.milvus.upsert(target_collection, rows)

        should_publish = publish and written_count == len(chunks)
        if should_publish:
            self.milvus.switch_alias(target_alias, target_collection)
        return MigrationManifest(
            ingestion_run_id=ingestion_run_id,
            source_path=self.chroma_path.as_posix(),
            source_collection=source_collection,
            target_collection=target_collection,
            target_alias=target_alias,
            data_version=data_version,
            source_hash=source_hash,
            source_count=len(records),
            normalized_count=len(chunks),
            written_count=written_count,
            issues=issues,
            published=should_publish,
        )

    def _read_source(self, collection_name: str) -> list[dict]:
        client = chromadb.PersistentClient(path=self.chroma_path.as_posix())
        collection = client.get_collection(collection_name)
        result = collection.get(include=["documents", "metadatas"])
        records = []
        for index, record_id in enumerate(result.get("ids", [])):
            records.append(
                {
                    "id": record_id,
                    "content": result["documents"][index],
                    "metadata": result["metadatas"][index] or {},
                }
            )
        return records

    @staticmethod
    def _source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
