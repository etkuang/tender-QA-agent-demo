# coding: utf-8
# @Author: Wang Qingkang

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import GenerationError, PlanningError, SQLValidationError, raise_model_error
from agent_layer.schemas import DependencyOutcome
from agent_layer.workflows.common import DomainProfile, format_dependency_outcomes
from agent_layer.data_domain.prompts import (
    DATA_ANSWER_PROMPT,
    DATA_INTENT_PROMPT,
    SQL_GENERATION_PROMPT,
    SQL_REPAIR_PROMPT,
)
from agent_layer.data_domain.schemas import (
    DataAnalysisSummary,
    DataContextBundle,
    DataIntentDecision,
    SQLCandidate,
    SQLRepairInput,
)

logger = get_logger("agent.data_domain.chains")


def format_domain_profile(profile: DomainProfile) -> str:
    return "\n".join(
        [
            f"类别：{profile.category.value}",
            f"名称：{profile.display_name}",
            f"能力说明：{profile.system_prompt}",
            f"SQL 视图：{', '.join(profile.sql_views) or '无'}",
            f"网站适配器：{', '.join(profile.website_adapters) or '无'}",
            f"必要证据字段：{', '.join(profile.required_evidence_fields) or '无'}",
            f"新鲜度规则：{profile.freshness_policy}",
            f"分析规则：{profile.analysis_template}",
            f"引用规则：{profile.citation_policy}",
        ]
    )


def format_data_context(context: DataContextBundle) -> str:
    tables = []
    for table in context.tables:
        columns = "\n".join(
            f"  - {column.name}: {column.description}; type={column.data_type or 'unknown'}; sensitive={column.is_sensitive}"
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
            f"approved={item.approved}"
        )
        for item in context.relationships
    ]
    metrics = [
        (
            f"{item.name}: {item.description}; formula={item.formula}; grain={item.grain}; "
            f"time_logic={item.time_logic or '无'}; caveats={'; '.join(item.caveats) or '无'}"
        )
        for item in context.metrics
    ]
    glossary = [
        f"{item.term}: {item.definition}; synonyms={', '.join(item.synonyms) or '无'}"
        for item in context.glossary_terms
    ]
    examples = [
        f"{item.name}: question={item.question}; sql={item.sql}; notes={item.notes or '无'}"
        for item in context.examples
    ]
    policy = context.access_policy
    return "\n\n".join(
        [
            f"dialect：{context.dialect}",
            f"allowed_tables：{', '.join(policy.allowed_tables)}",
            f"denied_columns：{', '.join(policy.denied_columns) or '无'}",
            f"max_rows：{policy.max_rows}",
            "tables：\n" + "\n\n".join(tables),
            "relationships：\n" + ("\n".join(relationships) or "无"),
            "metrics：\n" + ("\n".join(metrics) or "无"),
            "glossary：\n" + ("\n".join(glossary) or "无"),
            "approved_sql_examples：\n" + ("\n\n".join(examples) or "无"),
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


class DataIntentClassifier:
    def __init__(self, model: BaseChatModel, settings: Settings):
        structured = model.with_structured_output(
            DataIntentDecision,
            method="json_mode",
        )
        self.chain = DATA_INTENT_PROMPT | structured
        self.settings = settings

    async def classify(
        self,
        question: str,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        profile: DomainProfile,
        requires_fresh_data: bool,
    ) -> DataIntentDecision:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "profile": format_domain_profile(profile),
                    "question": question,
                    "dependency_outcomes": format_dependency_outcomes(dependency_outcomes),
                    "history": history,
                    "requires_fresh_data": requires_fresh_data,
                }
            )
        except Exception as exc:
            logger.exception("data intent classification failed | category=%s", profile.category.value)
            raise_model_error(exc, PlanningError)


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
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        context: DataContextBundle,
    ) -> SQLCandidate:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "dialect": context.dialect,
                    "intent": intent.model_dump_json(),
                    "context": format_data_context(context),
                    "question": question,
                    "dependency_outcomes": format_dependency_outcomes(dependency_outcomes),
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
        intent: DataIntentDecision,
        context: DataContextBundle,
    ) -> SQLCandidate:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "dialect": context.dialect,
                    "question": repair_input.original_question,
                    "intent": intent.model_dump_json(),
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


class DataAnswerChain:
    def __init__(self, model: BaseChatModel, settings: Settings):
        self.chain = DATA_ANSWER_PROMPT | model | StrOutputParser()
        self.settings = settings

    async def answer(
        self,
        profile: DomainProfile,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        context: DataContextBundle,
        sql: str,
        sql_result,
        analysis: DataAnalysisSummary | None,
        web_evidence: str,
        citation_feedback: str,
    ) -> str:
        try:
            return await self.chain.ainvoke(
                {
                    "profile": format_domain_profile(profile),
                    "question": question,
                    "intent": intent.model_dump_json(),
                    "dependency_outcomes": format_dependency_outcomes(dependency_outcomes),
                    "context": format_data_context(context),
                    "sql": sql,
                    "sql_result": format_sql_result(sql_result),
                    "analysis": format_analysis(analysis),
                    "web_evidence": web_evidence,
                    "citation_feedback": citation_feedback,
                }
            )
        except Exception as exc:
            logger.exception("data-domain answer generation failed | category=%s", profile.category.value)
            raise_model_error(exc, GenerationError)