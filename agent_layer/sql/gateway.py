# coding: utf-8

import json
import time
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import SQLValidationError
from agent_layer.schemas import DataResult, ResearchTask, SessionContext
from agent_layer.sql.schemas import SQLAuditEvent, SQLCandidate, ViewSchema
from agent_layer.sql.validator import SqlglotValidator

logger = get_logger("agent.sql.gateway")


SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你为只读分析视图生成单条 SELECT 或只读 CTE。
只能使用提供的视图和字段；禁止写操作、DDL、系统表、文件函数和未授权字段。
企业名称、地区、日期、型号等用户值使用命名参数，例如 :company_name，不直接拼接。
聚合、排名和金额必须遵守字段说明中的统计口径。不要添加解释文本。""",
        ),
        (
            "human",
            "研究任务：\n{task}\n\n允许视图：\n{schemas}\n\n上次校验错误：\n{feedback}\n\nJSON Schema：\n{output_schema}",
        ),
    ]
)


class SQLExecutor(Protocol):
    async def execute_read_only(
        self,
        statement: str,
        parameters: dict,
        timeout_seconds: float,
        max_rows: int,
        runtime_context: SessionContext,
    ) -> DataResult: ...


class SQLAuditSink(Protocol):
    async def record(self, event: SQLAuditEvent) -> None: ...


class SQLGateway(Protocol):
    async def execute(
        self,
        task: ResearchTask,
        allowed_views: list[str],
        runtime_context: SessionContext,
    ) -> DataResult: ...


class ReadOnlySQLGateway:
    def __init__(
        self,
        model: BaseChatModel,
        validator: SqlglotValidator,
        executor: SQLExecutor,
        schemas: dict[str, ViewSchema],
        settings: Settings,
        audit_sink: SQLAuditSink,
    ):
        structured = model.with_structured_output(
            SQLCandidate,
            method=settings.structured_output_method,
        )
        self.chain = SQL_GENERATION_PROMPT | structured
        self.validator = validator
        self.executor = executor
        self.schemas = schemas
        self.settings = settings
        self.audit_sink = audit_sink

    async def execute(
        self,
        task: ResearchTask,
        allowed_views: list[str],
        runtime_context: SessionContext,
    ) -> DataResult:
        selected_schemas = [self.schemas[name] for name in allowed_views if name in self.schemas]
        if len(selected_schemas) != len(allowed_views):
            raise SQLValidationError
        feedback = ""
        for attempt in range(2):
            candidate = await self._generate(task, selected_schemas, feedback)
            started = time.perf_counter()
            try:
                validated = self.validator.validate(candidate.statement, selected_schemas)
            except SQLValidationError:
                await self.audit_sink.record(
                    SQLAuditEvent(
                        task_id=task.task_id,
                        session_id=runtime_context.session_id or "",
                        statement=candidate.statement,
                        parameter_names=sorted(candidate.parameters),
                        status="validation_failed",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error_type="sql_validation_failed",
                    )
                )
                logger.warning("SQL validation failed | task_id=%s | attempt=%s", task.task_id, attempt + 1)
                feedback = "上一候选未通过只读、视图、字段、复杂度或 LIMIT 校验，请严格修正。"
                continue
            try:
                result = await self.executor.execute_read_only(
                    validated.statement,
                    candidate.parameters,
                    self.settings.sql_statement_timeout_seconds,
                    self.settings.sql_max_rows,
                    runtime_context,
                )
            except Exception as exc:
                await self.audit_sink.record(
                    SQLAuditEvent(
                        task_id=task.task_id,
                        session_id=runtime_context.session_id or "",
                        statement=validated.statement,
                        parameter_names=sorted(candidate.parameters),
                        status="execution_failed",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error_type=exc.__class__.__name__,
                    )
                )
                raise
            await self.audit_sink.record(
                SQLAuditEvent(
                    task_id=task.task_id,
                    session_id=runtime_context.session_id or "",
                    statement=validated.statement,
                    parameter_names=sorted(candidate.parameters),
                    status="completed",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    row_count=result.row_count,
                    query_id=result.query_id,
                )
            )
            return result
        raise SQLValidationError

    async def _generate(
        self,
        task: ResearchTask,
        schemas: list[ViewSchema],
        feedback: str,
    ) -> SQLCandidate:
        try:
            result = await self.chain.ainvoke(
                {
                    "task": task.model_dump_json(),
                    "schemas": json.dumps([schema.model_dump() for schema in schemas], ensure_ascii=False),
                    "feedback": feedback,
                    "output_schema": json.dumps(SQLCandidate.model_json_schema(), ensure_ascii=False),
                }
            )
        except Exception as exc:
            logger.exception("SQL generation failed | task_id=%s", task.task_id)
            raise SQLValidationError from exc
        if not isinstance(result, SQLCandidate):
            raise SQLValidationError
        return result
