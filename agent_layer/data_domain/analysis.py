# coding: utf-8
# @Author: Wang Qingkang

import hashlib
import json

from agent_layer.schemas import Category, DataResult, Evidence, SourceType
from agent_layer.data_domain.schemas import DataAnalysisSummary, DataContextBundle


class DataResultAnalyzer:
    def analyze(self, result: DataResult) -> DataAnalysisSummary:
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
            notes.append("结果已按配置的最大行数截断。")
        if result.row_count == 0:
            notes.append("SQL 在当前筛选条件下未返回数据。")
        return DataAnalysisSummary(
            query_id=result.query_id,
            row_count=result.row_count,
            truncated=result.truncated,
            columns=result.columns,
            numeric_metrics=numeric_metrics,
            missing_values={
                column: count
                for column, count in missing_values.items()
                if count
            },
            notes=notes,
        )


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


def _stable_id(namespace: str, primary: str, content: str) -> str:
    seed = f"{namespace}|{primary}|{content}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()