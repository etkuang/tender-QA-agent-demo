# coding: utf-8
# @Author: Wang Qingkang


from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeSearchRequest(BaseModel):
    index: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    law_name: str | None = None
    article_id: str | None = None
    as_of_date: date | None = None
    region: str | None = None


class KnowledgeHit(BaseModel):
    id: str
    text: str
    data: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float | None = None


class KnowledgeSearchResponse(BaseModel):
    index: str
    hits: list[KnowledgeHit] = Field(default_factory=list)