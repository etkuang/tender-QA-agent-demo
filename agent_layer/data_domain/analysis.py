# coding: utf-8
# @Author: Wang Qingkang

import hashlib
import json

from agent_layer.schemas import Category, DataResult, Evidence, SourceType
from agent_layer.data_domain.schemas import (
    DataAnalysisSummary,
    DataContextBundle,
    DataIntentDecision,
    DataQuestionType,
)


class DataResultAnalyzer:
    def analyze(
        self,
        result: DataResult,
        intent: DataIntentDecision,
    ) -> DataAnalysisSummary:
        numeric_metrics = {}
        missing_values = {column: 0 for column in result.columns}
        for row in result.rows:
            for column in result.columns:
                value = row.get(column)
                if value is None:
                    missing_values[column] += 1
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    bucket = numeric_metrics.setdefault(
                        column,
                        {"count": 0.0, "sum": 0.0, "min": value, "max": value},
                    )
                    bucket["count"] += 1
                    bucket["sum"] += value
                    bucket["min"] = min(bucket["min"], value)
                    bucket["max"] = max(bucket["max"], value)
        for bucket in numeric_metrics.values():
            if bucket["count"]:
                bucket["avg"] = bucket["sum"] / bucket["count"]
        notes = []
        if result.truncated:
            notes.append("Result was truncated by the configured row limit.")
        if result.row_count == 0:
            notes.append("SQL returned no rows for the selected filters.")
        return DataAnalysisSummary(
            query_id=result.query_id,
            question_type=intent.question_type,
            row_count=result.row_count,
            truncated=result.truncated,
            columns=result.columns,
            numeric_metrics=numeric_metrics,
            missing_values={column: count for column, count in missing_values.items() if count},
            notes=notes,
            chart_suggestion=self._chart_suggestion(result, intent),
        )

    @staticmethod
    def _chart_suggestion(
        result: DataResult,
        intent: DataIntentDecision,
    ) -> dict | None:
        if intent.requested_mode.value != "chart":
            return None
        if intent.question_type == DataQuestionType.TIME_SERIES:
            return {"type": "line", "x": result.columns[0] if result.columns else None, "y": result.columns[1:]}
        if intent.question_type in {DataQuestionType.RANKING, DataQuestionType.AGGREGATION, DataQuestionType.COMPARISON}:
            return {"type": "bar", "x": result.columns[0] if result.columns else None, "y": result.columns[1:]}
        return {"type": "table"}


def build_sql_evidence(
    result: DataResult,
    context: DataContextBundle,
    analysis: DataAnalysisSummary,
    statement: str,
    domain: Category,
    title: str,
) -> Evidence:
    content = json.dumps(
        {
            "sql": statement,
            "dialect": context.dialect,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "units": result.units,
            "filters": result.filters,
            "data_as_of": result.data_as_of.isoformat() if result.data_as_of else None,
            "query_id": result.query_id,
            "analysis": analysis.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    return Evidence(
        evidence_id=_stable_id(domain.value, result.query_id, content),
        domain=domain,
        source_type=SourceType.SQL,
        title=title,
        content=content,
        document_id=result.query_id,
        published_at=result.data_as_of,
        authority_level=3,
        freshness_level=2 if result.data_as_of else 1,
        metadata={
            "query_id": result.query_id,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "sql": statement,
            "analysis": analysis.model_dump(mode="json"),
        },
    )


def build_context_evidence(
    context: DataContextBundle,
    domain: Category,
    title: str,
) -> Evidence:
    content = context.model_dump_json()
    return Evidence(
        evidence_id=_stable_id(domain.value, "context", content),
        domain=domain,
        source_type=SourceType.SQL,
        title=title,
        content=content,
        authority_level=2,
        freshness_level=1,
        metadata={
            "context_tables": sorted(context.table_names),
            "metrics": [metric.name for metric in context.metrics],
            "glossary_terms": [term.term for term in context.glossary_terms],
        },
    )


def _stable_id(namespace: str, primary: str, content: str) -> str:
    seed = f"{namespace}|{primary}|{content}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()