# coding: utf-8
# @Author: Wang Qingkang

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_layer_old.schemas import Category


class DataColumnContext(BaseModel):
    name: str
    description: str
    data_type: str | None = None
    semantic_type: str | None = None
    is_sensitive: bool = False
    examples: list[str] = Field(default_factory=list)


class DataTableContext(BaseModel):
    name: str
    domain: Category
    description: str
    columns: list[DataColumnContext]
    primary_key: list[str] = Field(default_factory=list)
    default_time_column: str | None = None
    default_entity_column: str | None = None
    freshness: str | None = None

    @property
    def column_names(self) -> set[str]:
        return {column.name for column in self.columns}

    @property
    def sensitive_column_names(self) -> set[str]:
        return {column.name for column in self.columns if column.is_sensitive}


class DataRelationshipContext(BaseModel):
    left_table: str
    left_columns: list[str]
    right_table: str
    right_columns: list[str]
    relationship_type: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    description: str
    approved: bool = True


class DataMetricDefinition(BaseModel):
    name: str
    domain: Category
    description: str
    formula: str
    grain: str
    required_tables: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    time_logic: str | None = None
    caveats: list[str] = Field(default_factory=list)


class BusinessGlossaryTerm(BaseModel):
    term: str
    domain: Category
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class ApprovedSQLExample(BaseModel):
    name: str
    domain: Category
    question: str
    sql: str
    notes: str | None = None
    tables: list[str] = Field(default_factory=list)


class DataAccessPolicy(BaseModel):
    allowed_tables: list[str]
    denied_columns: list[str] = Field(default_factory=list)
    max_rows: int
    allow_select_star: bool = False
    require_limit: bool = True


class DataContextBundle(BaseModel):
    domain: Category
    dialect: str
    tables: list[DataTableContext]
    relationships: list[DataRelationshipContext] = Field(default_factory=list)
    metrics: list[DataMetricDefinition] = Field(default_factory=list)
    glossary_terms: list[BusinessGlossaryTerm] = Field(default_factory=list)
    examples: list[ApprovedSQLExample] = Field(default_factory=list)
    access_policy: DataAccessPolicy
    data_as_of: datetime | None = None

    @property
    def table_names(self) -> set[str]:
        return {table.name for table in self.tables}

    @property
    def column_names(self) -> set[str]:
        output = set()
        for table in self.tables:
            output.update(table.column_names)
        return output

    @property
    def sensitive_column_names(self) -> set[str]:
        output = set(self.access_policy.denied_columns)
        for table in self.tables:
            output.update(table.sensitive_column_names)
        return output


class SQLCandidate(BaseModel):
    statement: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class SQLValidationIssue(BaseModel):
    code: str
    message: str


class SQLValidationResult(BaseModel):
    valid: bool
    statement: str
    issues: list[SQLValidationIssue] = Field(default_factory=list)


class SQLExecutionRequest(BaseModel):
    statement: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float
    max_rows: int
    domain: Category
    request_id: str


class SQLRepairInput(BaseModel):
    original_question: str
    previous_sql: str
    validation_issues: list[SQLValidationIssue] = Field(default_factory=list)
    database_error: str | None = None
    attempt_index: int


class DataAuditEvent(BaseModel):
    request_id: str
    domain: Category
    statement: str | None = None
    parameter_names: list[str] = Field(default_factory=list)
    status: Literal["execution_failed", "completed"]
    duration_ms: float
    row_count: int | None = None
    query_id: str | None = None
    error_type: str | None = None


class DataAnalysisSummary(BaseModel):
    query_id: str
    row_count: int
    truncated: bool = False
    columns: list[str] = Field(default_factory=list)
    numeric_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    missing_values: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "ApprovedSQLExample",
    "BusinessGlossaryTerm",
    "DataAccessPolicy",
    "DataAnalysisSummary",
    "DataAuditEvent",
    "DataColumnContext",
    "DataContextBundle",
    "DataMetricDefinition",
    "DataRelationshipContext",
    "DataTableContext",
    "SQLCandidate",
    "SQLExecutionRequest",
    "SQLRepairInput",
    "SQLValidationIssue",
    "SQLValidationResult",
]