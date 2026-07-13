# coding: utf-8
# @Author: Wang Qingkang

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from agent_layer.config import Settings
from agent_layer.errors import SQLExecutionError
from agent_layer.schemas import Category, DataResult, Evidence, PolicyQuery, SourceTier, ToolEvent
from agent_layer.workflows.common import ToolName
from agent_layer.data_domain.analysis import DataResultAnalyzer, build_sql_evidence
from agent_layer.data_domain.chains import DataSQLGenerator, DataSQLRepairChain
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.schemas import (
    DataAnalysisSummary,
    DataContextBundle,
    SQLExecutionRequest,
    SQLRepairInput,
    SQLValidationResult,
)
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.retrieval.pipeline import RetrievalPipeline


@dataclass(frozen=True)
class ChildTaskToolRequest:
    category: Category
    query: str
    run_id: str
    progress_callback: Callable[[ToolEvent], Awaitable[None]]
    policy_query: PolicyQuery | None = None
    keywords: list[str] = field(default_factory=list)
    required_source_tier: SourceTier | None = None


@dataclass
class ChildTaskToolResult:
    evidence: list[Evidence] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    context: DataContextBundle | None = None
    sql_statement: str | None = None
    sql_result: DataResult | None = None
    analysis: DataAnalysisSummary | None = None
    used_model_only: bool = False
    model_calls: int = 0


class ChildTaskTool(Protocol):
    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult: ...


class ChildTaskToolRegistry:
    def __init__(self, tools: dict[ToolName, ChildTaskTool]):
        self.tools = tools

    def subset(self, tool_names: list[ToolName]) -> "ChildTaskToolRegistry":
        return ChildTaskToolRegistry(
            {tool_name: self.tools[tool_name] for tool_name in tool_names}
        )

    async def invoke(
        self,
        tool_name: ToolName,
        request: ChildTaskToolRequest,
    ) -> ChildTaskToolResult:
        return await self.tools[tool_name].invoke(request)

    async def invoke_tier(
        self,
        tool_names: list[ToolName],
        request: ChildTaskToolRequest,
    ) -> list[ChildTaskToolResult]:
        return await asyncio.gather(
            *(self.tools[tool_name].invoke(request) for tool_name in tool_names)
        )


class RAGChildTaskTool:
    def __init__(self, retrieval: RetrievalPipeline):
        self.retrieval = retrieval

    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult:
        output = await self.retrieval.retrieve_policy(
            request.query,
            request.policy_query,
        )
        for event in output.events:
            await request.progress_callback(event)
        return ChildTaskToolResult(
            evidence=output.evidence,
            tool_events=output.events,
        )


class SQLChildTaskTool:
    def __init__(
        self,
        sql_generator: DataSQLGenerator,
        sql_repair_chain: DataSQLRepairChain,
        sql_validator: SQLPolicyValidator,
        sql_gateway: SQLGateway | None,
        context_retriever: DataContextRetriever,
        analyzer: DataResultAnalyzer,
        settings: Settings,
    ):
        self.sql_generator = sql_generator
        self.sql_repair_chain = sql_repair_chain
        self.sql_validator = sql_validator
        self.sql_gateway = sql_gateway
        self.context_retriever = context_retriever
        self.analyzer = analyzer
        self.settings = settings

    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult:
        context = self.context_retriever.retrieve(
            request.query,
            request.category,
        )
        if self.sql_gateway is None:
            event = ToolEvent(
                stage="sql",
                status="skipped",
                summary="未配置只读 SQL 网关。",
            )
            await request.progress_callback(event)
            return ChildTaskToolResult(tool_events=[event], context=context)

        start_event = ToolEvent(
            stage="sql",
            status="started",
            summary="正在生成并执行只读 SQL 查询。",
        )
        await request.progress_callback(start_event)
        started = time.perf_counter()
        candidate = await self.sql_generator.generate(
            request.query,
            context,
        )
        model_calls = 1
        validation = SQLValidationResult(valid=False, statement=candidate.statement)
        database_error = None

        for attempt in range(self.settings.sql_repair_attempts + 1):
            validation = self.sql_validator.validate(candidate, context)
            if validation.valid:
                try:
                    sql_result = await self.sql_gateway.execute(
                        SQLExecutionRequest(
                            statement=validation.statement,
                            parameters=candidate.parameters,
                            timeout_seconds=self.settings.sql_statement_timeout_seconds,
                            max_rows=self.settings.sql_max_rows,
                            domain=request.category,
                            request_id=request.run_id,
                        )
                    )
                except SQLExecutionError as exc:
                    database_error = exc.user_message
                else:
                    analysis = self.analyzer.analyze(sql_result)
                    evidence = build_sql_evidence(
                        sql_result,
                        context,
                        analysis,
                        validation.statement,
                        request.category,
                        "结构化数据库查询结果",
                    )
                    event = ToolEvent(
                        stage="sql",
                        status="completed",
                        summary=f"只读 SQL 查询完成，返回 {sql_result.row_count} 行。",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        details={"row_count": sql_result.row_count},
                    )
                    await request.progress_callback(event)
                    return ChildTaskToolResult(
                        evidence=[evidence],
                        tool_events=[start_event, event],
                        context=context,
                        sql_statement=validation.statement,
                        sql_result=sql_result,
                        analysis=analysis,
                        model_calls=model_calls,
                    )
            if attempt >= self.settings.sql_repair_attempts:
                break
            candidate = await self.sql_repair_chain.repair(
                SQLRepairInput(
                    original_question=request.query,
                    previous_sql=candidate.statement,
                    validation_issues=validation.issues,
                    database_error=database_error,
                    attempt_index=attempt + 1,
                ),
                context,
            )
            model_calls += 1

        event = ToolEvent(
            stage="sql",
            status="failed",
            summary=database_error or "生成的 SQL 未通过只读安全校验。",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        await request.progress_callback(event)
        return ChildTaskToolResult(
            tool_events=[start_event, event],
            context=context,
            sql_statement=candidate.statement,
            model_calls=model_calls,
        )


class WebsiteChildTaskTool:
    def __init__(self, supplementer: WebsiteSupplementer):
        self.supplementer = supplementer

    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult:
        evidence, events, _ = await self.supplementer.search(
            request.query,
            request.category,
            request.keywords,
            request.required_source_tier,
            request.progress_callback,
        )
        return ChildTaskToolResult(evidence=evidence, tool_events=events)


class ModelOnlyChildTaskTool:
    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult:
        event = ToolEvent(
            stage="model_only",
            status="completed",
            summary="未使用外部证据，进入明确标注的模型知识回答流程。",
        )
        await request.progress_callback(event)
        return ChildTaskToolResult(
            tool_events=[event],
            used_model_only=True,
        )