# coding: utf-8

import hashlib
from datetime import datetime

from agent_layer_old.schemas import Category, Evidence, SearchResult, SourceTier, SourceType


class EvidenceAdapter:
    def from_policy_chunk(self, chunk: dict) -> Evidence:
        metadata = chunk.get("data", {})
        law_name = metadata.get("law_name") or metadata.get("doc_title") or metadata.get("source")
        article_id = metadata.get("article_id") or metadata.get("article_num")
        document_id = chunk.get("id") or metadata.get("document_id")
        title = law_name or "未知政策来源"
        if article_id not in (None, "", "unknown", "full"):
            title = f"{title} 第{article_id}条"
        freshness_level = (
            metadata["freshness_level"]
            if "freshness_level" in metadata
            else self._freshness_level(metadata)
        )
        return Evidence(
            evidence_id=self._stable_id("policy", document_id, chunk.get("text", "")),
            domain=Category.POLICY,
            source_type=SourceType.LOCAL_DOCUMENT,
            title=title,
            content=chunk.get("text", ""),
            url=metadata.get("source_url") or metadata.get("url"),
            document_id=document_id,
            law_name=law_name,
            article_id=f"{article_id}" if article_id not in (None, "") else None,
            published_at=self._parse_datetime(metadata.get("published_at") or metadata.get("publish_date")),
            score=chunk.get("score"),
            authority_level=metadata.get("authority_level") or 0,
            freshness_level=freshness_level,
            metadata=self._normalized_metadata(metadata, chunk),
        )

    def from_search_result(self, result: SearchResult, category: Category) -> Evidence:
        authority = {
            SourceTier.OFFICIAL: 3,
            SourceTier.AUTHORITATIVE: 2,
            SourceTier.GENERAL: 1,
            SourceTier.UNTRUSTED: 0,
        }[result.source_tier]
        return Evidence(
            evidence_id=self._stable_id(category.value, result.content_hash, result.url),
            domain=category,
            source_type=SourceType.WEBSITE,
            title=result.title,
            content=result.snippet,
            url=result.url,
            published_at=result.published_at,
            retrieved_at=result.retrieved_at,
            authority_level=authority,
            freshness_level=2 if result.published_at else 1,
            metadata={
                "source_name": result.source_name,
                "source_tier": result.source_tier.value,
                "content_hash": result.content_hash,
                **result.structured_data,
            },
        )

    @staticmethod
    def _stable_id(namespace: str, primary, content: str) -> str:
        seed = f"{namespace}|{primary or ''}|{content}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_datetime(value) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _freshness_level(metadata: dict) -> int:
        status = metadata.get("validity_status")
        if status == "effective":
            return 3
        if status in {"amended", "repealed"}:
            return 0
        return 1

    @staticmethod
    def _normalized_metadata(metadata: dict, chunk: dict) -> dict:
        return {
            **metadata,
            "vector_rank": chunk.get("vector_rank"),
            "bm25_rank": chunk.get("bm25_rank"),
            "fusion_score": chunk.get("fusion_score", chunk.get("score")),
        }
