# coding: utf-8
# @Author: Wang Qingkang

import json

from langchain_core.language_models import BaseChatModel

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import SQLValidationError, raise_model_error
from agent_layer.data_domain.prompts import SQL_GENERATION_PROMPT, SQL_REPAIR_PROMPT
from agent_layer.data_domain.schemas import (
    DataAnalysisSummary,
    DataContextBundle,
    SQLCandidate,
    SQLRepairInput,
)

logger = get_logger("agent.data_domain.chains")



def format_data_context(context: DataContextBundle) -> str:
    tables = []
    for table in context.tables:
        columns = "\n".join(
            f"  - {column.name}: {column.description}; 类型={column.data_type or '未知'}; 敏感={column.is_sensitive}"
            for column in table.columns
        )
        tables.append(
            "\n".join(
                [
                    f"表：{table.name}",
                    f"领域：{table.domain.value}",
                    f"说明：{table.description}",
                    f"主键：{', '.join(table.primary_key) or '无'}",
                    f"默认时间字段：{table.default_time_column or '无'}",
                    "字段：",
                    columns,
                ]
            )
        )
    relationships = [
        (
            f"{item.left_table}({', '.join(item.left_columns)}) -> "
            f"{item.right_table}({', '.join(item.right_columns)}): {item.description}; "
            f"已批准={item.approved}"
        )
        for item in context.relationships
    ]
    metrics = [
        (
            f"{item.name}: {item.description}; 公式={item.formula}; 粒度={item.grain}; "
            f"时间规则={item.time_logic or '无'}; 注意事项={'; '.join(item.caveats) or '无'}"
        )
        for item in context.metrics
    ]
    glossary = [
        f"{item.term}: {item.definition}; 同义词={', '.join(item.synonyms) or '无'}"
        for item in context.glossary_terms
    ]
    examples = [
        f"{item.name}: 问题={item.question}; SQL={item.sql}; 备注={item.notes or '无'}"
        for item in context.examples
    ]
    policy = context.access_policy
    return "\n\n".join(
        [
            f"SQL 方言：{context.dialect}",
            f"允许访问的表：{', '.join(policy.allowed_tables)}",
            f"禁止访问的字段：{', '.join(policy.denied_columns) or '无'}",
            f"最大返回行数：{policy.max_rows}",
            "数据表：\n" + "\n\n".join(tables),
            "关系：\n" + ("\n".join(relationships) or "无"),
            "指标：\n" + ("\n".join(metrics) or "无"),
            "业务词汇：\n" + ("\n".join(glossary) or "无"),
            "已批准的 SQL 示例：\n" + ("\n\n".join(examples) or "无"),
        ]
    )


def format_sql_result(result) -> str:
    if result is None:
        return "无 SQL 执行结果。"
    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
    )


def format_analysis(analysis: DataAnalysisSummary | None) -> str:
    if analysis is None:
        return "无确定性分析摘要。"
    return analysis.model_dump_json()


def format_validation_issues(issues) -> str:
    if not issues:
        return "无。"
    return "\n".join(f"- {item.code}: {item.message}" for item in issues)


class DataSQLGenerator:
    def __init__(self, model: BaseChatModel, settings: Settings):
        structured = model.with_structured_output(
            SQLCandidate,
            method="json_mode",
        )
        self.chain = SQL_GENERATION_PROMPT | structured
        self.settings = settings

    async def generate(
        self,
        query: str,
        context: DataContextBundle,
    ) -> SQLCandidate:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "dialect": context.dialect,
                    "query": query,
                    "context": format_data_context(context),
                }
            )
        except Exception as exc:
            logger.exception("SQL generation failed | category=%s", context.domain.value)
            raise_model_error(exc, SQLValidationError)


class DataSQLRepairChain:
    def __init__(self, model: BaseChatModel, settings: Settings):
        structured = model.with_structured_output(
            SQLCandidate,
            method="json_mode",
        )
        self.chain = SQL_REPAIR_PROMPT | structured
        self.settings = settings

    async def repair(
        self,
        repair_input: SQLRepairInput,
        context: DataContextBundle,
    ) -> SQLCandidate:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "dialect": context.dialect,
                    "query": repair_input.original_question,
                    "context": format_data_context(context),
                    "previous_sql": repair_input.previous_sql,
                    "validation_issues": format_validation_issues(repair_input.validation_issues),
                    "database_error": repair_input.database_error or "无。",
                    "attempt_index": repair_input.attempt_index,
                }
            )
        except Exception as exc:
            logger.exception("SQL repair failed | category=%s", context.domain.value)
            raise_model_error(exc, SQLValidationError)
