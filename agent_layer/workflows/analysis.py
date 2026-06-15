# coding: utf-8

from agent_layer.schemas import AnalysisSummary, DataResult


class DatasetAnalyzer:
    """Deterministic summary generator; it never executes model-provided code."""

    def analyze(self, result: DataResult, analysis_template: str) -> AnalysisSummary:
        numeric_metrics = {}
        missing_values = {}
        for column in result.columns:
            values = [row.get(column) for row in result.rows]
            missing_values[column] = sum(value is None for value in values)
            numeric_values = [
                value
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric_values:
                numeric_metrics[column] = {
                    "minimum": min(numeric_values),
                    "maximum": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                }
        notes = [analysis_template]
        if result.truncated:
            notes.append("结果已按行数上限截断，统计摘要仅基于返回数据。")
        return AnalysisSummary(
            query_id=result.query_id,
            sample_size=len(result.rows),
            filters=result.filters,
            numeric_metrics=numeric_metrics,
            missing_values=missing_values,
            notes=notes,
        )
