# coding: utf-8

from pydantic import BaseModel, Field

from knowledge_base_layer.embeddings import EmbeddingGateway
from knowledge_base_layer.ingestion.normalizer import ChunkNormalizer, NormalizationIssue, NormalizedChunk
from knowledge_base_layer.retrieval.milvus import MilvusStore


class IngestionManifest(BaseModel):
    ingestion_run_id: str
    collection: str
    alias: str
    data_version: str
    source_count: int
    written_count: int
    issues: list[NormalizationIssue] = Field(default_factory=list)
    published: bool = False


class PolicyIngestionPipeline:
    def __init__(
        self,
        milvus: MilvusStore,
        embeddings: EmbeddingGateway,
        normalizer: ChunkNormalizer | None = None,
    ):
        self.milvus = milvus
        self.embeddings = embeddings
        self.normalizer = normalizer or ChunkNormalizer()

    def ingest(
        self,
        chunks: list[NormalizedChunk],
        alias: str,
        data_version: str,
        ingestion_run_id: str,
        batch_size: int = 128,
        publish: bool = False,
    ) -> IngestionManifest:
        issues = []
        ids = set()
        for chunk in chunks:
            issues.extend(self.normalizer.validate(chunk))
            if chunk.id in ids:
                issues.append(
                    NormalizationIssue(
                        record_id=chunk.id,
                        field="id",
                        message="主键重复。",
                        critical=True,
                    )
                )
            ids.add(chunk.id)
        if any(issue.critical for issue in issues):
            return IngestionManifest(
                ingestion_run_id=ingestion_run_id,
                collection=f"{alias}_{data_version}",
                alias=alias,
                data_version=data_version,
                source_count=len(chunks),
                written_count=0,
                issues=issues,
                published=False,
            )

        collection = f"{alias}_{data_version}"
        self.milvus.create_version(collection)
        written_count = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_documents([chunk.content for chunk in batch])
            rows = [
                chunk.model_copy(update={"dense_vector": vector, "data_version": data_version}).model_dump()
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            written_count += self.milvus.upsert(collection, rows)
        should_publish = publish and written_count == len(chunks)
        if should_publish:
            self.milvus.switch_alias(alias, collection)
        return IngestionManifest(
            ingestion_run_id=ingestion_run_id,
            collection=collection,
            alias=alias,
            data_version=data_version,
            source_count=len(chunks),
            written_count=written_count,
            issues=issues,
            published=should_publish,
        )
