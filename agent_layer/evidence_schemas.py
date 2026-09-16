# coding: utf-8
# @Author: Wang Qingkang

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from agent_layer.config import settings


class EvidenceSourceType(StrEnum):
    WEBSITE = "website"


class EvidenceDraft(BaseModel):
    """Common model-generated fields; use a source-specific draft subtype."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=settings.evidence_title_max_chars,
    )
    content: str = Field(
        min_length=1,
        max_length=settings.evidence_content_max_chars,
    )
    source_type: EvidenceSourceType
    updated_at: datetime | None = None
    reliability_description: str = Field(
        min_length=1,
        max_length=settings.evidence_reliability_max_chars,
    )


class WebsiteEvidenceDraft(EvidenceDraft):
    """Model-generated website evidence before id assignment."""

    source_type: Literal[EvidenceSourceType.WEBSITE]
    url: AnyHttpUrl


class WebsiteEvidenceDraftList(BaseModel):
    """Multiple coherent evidence records generated from website material."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[WebsiteEvidenceDraft]


class WebsiteEvidence(WebsiteEvidenceDraft):
    """Stored website evidence with a Milvus-assigned ID and session ownership."""

    evidence_id: str = Field(min_length=1)
    session_id: str