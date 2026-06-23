# coding: utf-8

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, RootModel, model_validator


class Category(StrEnum):
    POLICY = "policy"
    TENDER = "tender"
    PUBLIC_OPINION = "public_opinion"
    COMPANY = "company"
    PRICE = "price"
    PRODUCT = "product"
    OTHER = "other"
    UNCLEAR = "unclear"


class SourceType(StrEnum):
    LOCAL_DOCUMENT = "local_document"
    SQL = "sql"
    WEBSITE = "website"


class SourceTier(StrEnum):
    OFFICIAL = "official"
    AUTHORITATIVE = "authoritative"
    GENERAL = "general"
    UNTRUSTED = "untrusted"


class StreamEventType(StrEnum):
    ROUTE = "route"
    PROGRESS = "progress"
    REASONING_SUMMARY = "reasoning_summary"
    SOURCE = "source"
    ASSISTANT_DELTA = "assistant_delta"
    ERROR = "error"


class QuickResponseType(StrEnum):
    GREETING_ZH = "greeting_zh"
    GREETING_EN = "greeting_en"
    THANKS_ZH = "thanks_zh"
    THANKS_EN = "thanks_en"
    GOODBYE_ZH = "goodbye_zh"
    GOODBYE_EN = "goodbye_en"
    CAPABILITIES_ZH = "capabilities_zh"
    CAPABILITIES_EN = "capabilities_en"
    WELLBEING_ZH = "wellbeing_zh"
    WELLBEING_EN = "wellbeing_en"
    ACKNOWLEDGEMENT_ZH = "acknowledgement_zh"
    ACKNOWLEDGEMENT_EN = "acknowledgement_en"
    COMPLIMENT_ZH = "compliment_zh"
    COMPLIMENT_EN = "compliment_en"
    APOLOGY_ZH = "apology_zh"
    APOLOGY_EN = "apology_en"
    NONE = "none"


class ChildTask(BaseModel):
    task_id: str
    question: str
    category: Category
    depends_on: list[str] = Field(default_factory=list)
    requires_fresh_data: bool = False
    clarification_question: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_clarification(self) -> Self:
        has_clarification = self.clarification_question is not None
        if (self.category == Category.UNCLEAR) != has_clarification:
            raise ValueError("Only unclear child tasks must provide clarification_question")
        return self


class ChildTaskList(RootModel[list[ChildTask]]):
    root: list[ChildTask] = Field(min_length=1)


class RoutePlan(BaseModel):
    action: Literal["execute", "clarify", "general_answer"]
    tasks: list[ChildTask]
    reason: str


class QuickResponseDecision(BaseModel):
    intent: QuickResponseType
    reason: str = Field(min_length=1)


class Evidence(BaseModel):
    evidence_id: str
    domain: Category
    source_type: SourceType
    title: str
    content: str
    url: str | None = None
    document_id: str | None = None
    law_name: str | None = None
    article_id: str | None = None
    entity_id: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score: float | None = None
    authority_level: int = Field(default=0, ge=0, le=3)
    freshness_level: int = Field(default=0, ge=0, le=3)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    citation_id: str
    evidence_id: str
    label: str
    url: str | None = None


class RetrievalAssessment(BaseModel):
    sufficient: bool
    reason: str
    missing_information: list[str] = Field(default_factory=list)
    covered_claims: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    need_more_local_retrieval: bool = False
    need_official_web_search: bool = False
    follow_up_queries: list[str] = Field(default_factory=list)
    usable_evidence_ids: list[str] = Field(default_factory=list)


class PolicyQuery(BaseModel):
    law_name: str | None = None
    article_id: str | None = None
    as_of_date: date | None = None
    region: str | None = None


class TimeRange(BaseModel):
    start: date | None = None
    end: date | None = None


class ResearchTask(BaseModel):
    task_id: str
    goal: str
    preferred_source: Literal["sql", "website", "both"]
    domain: Category | None = None
    depends_on: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    subject: str
    keywords: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    region: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)


class DataResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    units: dict[str, str] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    data_as_of: datetime | None = None
    query_id: str


class AnalysisSummary(BaseModel):
    query_id: str
    sample_size: int
    filters: dict[str, Any] = Field(default_factory=dict)
    numeric_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    missing_values: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class WebsiteQuery(BaseModel):
    query: str
    category: Category
    keywords: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    region: str | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    source_name: str
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snippet: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    source_tier: SourceTier
    content_hash: str


class ToolEvent(BaseModel):
    stage: str
    status: Literal["started", "completed", "skipped", "failed"]
    summary: str
    duration_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    model_calls: int = 0
    run_id: str | None = None


class StreamEvent(BaseModel):
    type: StreamEventType
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
