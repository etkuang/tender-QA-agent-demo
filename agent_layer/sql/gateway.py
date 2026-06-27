# coding: utf-8

import json
import time
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger, get_request_id
from agent_layer.config import Settings
from agent_layer.errors import SQLValidationError
from agent_layer.schemas import DataResult, ResearchTask
from agent_layer.sql.schemas import SQLAuditEvent, SQLCandidate, ViewSchema
from agent_layer.sql.validator import SqlglotValidator

logger = get_logger("agent.sql.gateway")


SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是只读 SQL 候选生成器。
根据研究任务和允许视图生成一个 SQLCandidate。

只输出 JSON，不要输出解释或多余文本。
格式示例：
{
  "statement": "SELECT column_name FROM view_name WHERE field = :value LIMIT 200",
  "parameters": {"value": "参数值"},
  "selected_views": ["view_name"]
}

字段说明：
- statement：单条 SELECT 或只读 CTE，不生成写操作、DDL、系统表、文件函数或未授权字段。
- parameters：用户给出的企业名称、地区、日期、型号等值必须使用命名参数，不直接拼接到 SQL 字符串。
- selected_views：statement 实际使用的视图名称。

SQL 规则：
- 只能使用允许视图和字段。
- 聚合、排名和金额必须遵守字段说明中的统计口径。
- 上次校验错误不为空时，根据错误修正 SQL。""",
        ),
        (
            "human",
            "研究任务：\n{task}\n\n允许视图：\n{schemas}\n\n上次校验错误：\n{feedback}",
        ),
    ]
)


def _format_view_schemas(schemas: list[ViewSchema]) -> str:
    blocks = []
    for schema in schemas:
        columns = "\n".join(
            f"- {name}: {description}"
            for name, description in schema.columns.items()
        )
        blocks.append(
            "\n".join(
                [
                    f"视图：{schema.name}",
                    f"说明：{schema.description}",
                    "字段：",
                    columns,
                ]
            )
        )
    return "\n\n".join(blocks)


class SQLExecutor(Protocol):
    async def execute_read_only(
        self,
        statement: str,
        parameters: dict,
        timeout_seconds: float,
        max_rows: int,
    ) -> DataResult: ...


class SQLAuditSink(Protocol):
    async def record(self, event: SQLAuditEvent) -> None: ...


class SQLGateway(Protocol):
    async def execute(
        self,
        task: ResearchTask,
        allowed_views: list[str],
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
            method="json_mode",
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
                        request_id=get_request_id(),
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
                )
            except Exception as exc:
                await self.audit_sink.record(
                    SQLAuditEvent(
                        task_id=task.task_id,
                        request_id=get_request_id(),
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
                    request_id=get_request_id(),
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
            result = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "task": task.model_dump_json(),
                    "schemas": _format_view_schemas(schemas),
                    "feedback": feedback,
                }
            )
        except Exception as exc:
            logger.exception("SQL generation failed | task_id=%s", task.task_id)
            raise SQLValidationError from exc
        return result
