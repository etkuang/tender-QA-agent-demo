# coding: utf-8

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, Field


class NormalizedChunk(BaseModel):
    id: str
    content: str
    title: str = ""
    source_url: str = ""
    document_id: str = ""
    law_name: str = ""
    article_id: str = ""
    entity_id: str = ""
    published_at: str = ""
    authority_level: int = 0
    freshness_level: int = 0
    parent_id: str = ""
    chunk_type: str = "document"
    validity_status: str = "unknown"
    region: str = "national"
    effective_date: str = ""
    end_date: str = ""
    data_version: str
    metadata: str = "{}"
    dense_vector: list[float] = Field(default_factory=list)


class NormalizationIssue(BaseModel):
    record_id: str
    field: str
    message: str
    critical: bool


class ChunkNormalizer:
    allowed_validity = {"draft", "effective", "amended", "repealed", "unknown"}

    def normalize(
        self,
        record_id: str,
        content: str,
        metadata: dict,
        data_version: str,
    ) -> tuple[NormalizedChunk, list[NormalizationIssue]]:
        law_name = metadata.get("law_name") or metadata.get("doc_title") or metadata.get("source") or ""
        article_id = metadata.get("article_id") or metadata.get("article_num") or ""
        chunk_type = metadata.get("chunk_type") or metadata.get("type") or "document"
        parent_id = metadata.get("parent_id") or ""
        validity = metadata.get("validity_status") or "unknown"
        if validity not in self.allowed_validity:
            validity = "unknown"
        stable_id = record_id or self._stable_id(law_name, article_id, content, data_version)
        normalized = NormalizedChunk(
            id=stable_id,
            content=content,
            title=metadata.get("title") or metadata.get("article") or law_name,
            source_url=metadata.get("source_url") or metadata.get("url") or "",
            document_id=metadata.get("document_id") or stable_id,
            law_name=law_name,
            article_id=f"{article_id}" if article_id else "",
            entity_id=metadata.get("entity_id") or metadata.get("company_id") or "",
            published_at=self._date_text(metadata.get("published_at") or metadata.get("publish_date")),
            authority_level=metadata.get("authority_level") or 0,
            freshness_level=metadata.get("freshness_level") or 0,
            parent_id=parent_id,
            chunk_type=chunk_type,
            validity_status=validity,
            region=metadata.get("region") or "national",
            effective_date=self._date_text(metadata.get("effective_date")),
            end_date=self._date_text(metadata.get("end_date")),
            data_version=data_version,
            metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        )
        return normalized, self.validate(normalized)

    @staticmethod
    def validate(chunk: NormalizedChunk) -> list[NormalizationIssue]:
        issues = []
        if not chunk.content.strip():
            issues.append(NormalizationIssue(record_id=chunk.id, field="content", message="正文为空。", critical=True))
        if chunk.chunk_type in {"parent", "child"} and not chunk.law_name:
            issues.append(NormalizationIssue(record_id=chunk.id, field="law_name", message="法条缺少法规名称。", critical=True))
        if chunk.chunk_type in {"parent", "child"} and not chunk.article_id:
            issues.append(NormalizationIssue(record_id=chunk.id, field="article_id", message="法条缺少条号。", critical=True))
        if chunk.chunk_type == "child" and not chunk.parent_id:
            issues.append(NormalizationIssue(record_id=chunk.id, field="parent_id", message="child 缺少 parent_id。", critical=True))
        if chunk.chunk_type in {"parent", "child"} and chunk.validity_status == "unknown":
            issues.append(NormalizationIssue(record_id=chunk.id, field="validity_status", message="法规效力未知。", critical=True))
        return issues

    @staticmethod
    def _stable_id(law_name: str, article_id, content: str, data_version: str) -> str:
        raw = f"{law_name}|{article_id}|{content}|{data_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _date_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        return f"{value}"
