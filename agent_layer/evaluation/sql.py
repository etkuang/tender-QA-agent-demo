# coding: utf-8

from pydantic import BaseModel, Field

from agent_layer.schemas import ResearchTask, SessionContext
from agent_layer.sql.gateway import SQLGateway


class SQLEvaluationSample(BaseModel):
    sample_id: str
    task: ResearchTask
    allowed_views: list[str]
    expected_columns: list[str] = Field(default_factory=list)
    expected_row_count: int | None = None


class SQLEvaluation(BaseModel):
    total: int
    execution_success_rate: float
    result_correct_rate: float
    failures: list[str]


async def evaluate_sql(
    gateway: SQLGateway,
    samples: list[SQLEvaluationSample],
) -> SQLEvaluation:
    execution_success = 0
    result_correct = 0
    failures = []
    for sample in samples:
        try:
            result = await gateway.execute(sample.task, sample.allowed_views, SessionContext())
        except Exception:
            failures.append(sample.sample_id)
            continue
        execution_success += 1
        columns_match = not sample.expected_columns or result.columns == sample.expected_columns
        rows_match = sample.expected_row_count is None or result.row_count == sample.expected_row_count
        if columns_match and rows_match:
            result_correct += 1
        else:
            failures.append(sample.sample_id)
    total = len(samples)
    return SQLEvaluation(
        total=total,
        execution_success_rate=execution_success / total if total else 0,
        result_correct_rate=result_correct / total if total else 0,
        failures=failures,
    )
