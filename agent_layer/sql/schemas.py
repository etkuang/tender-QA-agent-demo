# coding: utf-8

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_layer.schemas import DataResult


class SQLCandidate(BaseModel):
    statement: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    selected_views: list[str] = Field(default_factory=list)


class SQLValidationResult(BaseModel):
    statement: str
    tables: list[str]
    columns: list[str]
    limit: int


class ViewSchema(BaseModel):
    name: str
    description: str
    columns: dict[str, str]
    sensitive_columns: list[str] = Field(default_factory=list)


class SQLAuditEvent(BaseModel):
    task_id: str
    request_id: str
    statement: str
    parameter_names: list[str]
    status: Literal["validation_failed", "execution_failed", "completed"]
    duration_ms: float
    row_count: int | None = None
    query_id: str | None = None
    error_type: str | None = None


__all__ = [
    "DataResult",
    "SQLCandidate",
    "SQLValidationResult",
    "SQLAuditEvent",
    "ViewSchema",
]
