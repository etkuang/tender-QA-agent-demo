# coding: utf-8
# @Author: Wang Qingkang

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from agent_layer.config import Settings
from agent_layer.errors import SQLExecutionError
from agent_layer.schemas import Category, Evidence, PolicyQuery, SourceTier, ToolEvent
from agent_layer.data_domain.analysis import DataResultAnalyzer, build_sql_evidence
from agent_layer.data_domain.chains import DataSQLGenerator, DataSQLRepairChain
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.schemas import (
    SQLExecutionRequest,
    SQLRepairInput,
    SQLValidationResult,
)
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.workflows.common import (
    ChildTaskQueryResult,
    ChildTaskToolSpec,
    ToolName,
)


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
    query_results: list[ChildTaskQueryResult] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    model_calls: int = 0

    def collect_evidence(self) -> list[Evidence]:
        return [
            evidence
            for query_result in self.query_results
            for evidence in query_result.query_result
        ]


class ChildTaskTool(Protocol):
    async def invoke(self, request: ChildTaskToolRequest) -> ChildTaskToolResult: ...


class ChildTaskToolRegistry:
    tool_pool = {}
    tool_specs = {
        "rag": ChildTaskToolSpec(
            tool_name="rag",
            description="检索本地政策知识库中的政策条文和政策文档证据。",
            query_format="生成适合语义检索的独立查询，保留政策名称、条款、地区、时间和需要核验的事实。",
        ),
        "sql": ChildTaskToolSpec(
            tool_name="sql",
            description="查询已配置的结构化业务数据库，并对查询结果执行确定性分析。",
            query_format="生成面向 SQL 生成器的完整自然语言查询要求，明确实体、字段、筛选条件、时间范围、指标、聚合、排序和返回数量；不得直接生成 SQL。",
        ),
        "website": ChildTaskToolSpec(
            tool_name="website",
            description="检索当前类别的公开网站证据，用于核验当前信息或数据库之外的信息。",
            query_format="生成适合网站检索的独立查询，明确实体、关键词、地区、时间范围、需要核验的事实和来源要求。",
        ),
    }

    @classmethod
    def initialize(cls, tools: dict[ToolName, ChildTaskTool]) -> None:
        cls.tool_pool = tools

    @classmethod
    def get_specs(cls, tool_names: list[ToolName]) -> list[ChildTaskToolSpec]:
        return [cls.tool_specs[tool_name] for tool_name in tool_names]

    @classmethod
    async def invoke(
        cls,
        tool_name: ToolName,
        request: ChildTaskToolRequest,
    ) -> ChildTaskToolResult:
        return await cls.tool_pool[tool_name].invoke(request)

    @classmethod
    async def invoke_tier(
        cls,
        requests: list[tuple[ToolName, ChildTaskToolRequest]],
    ) -> list[ChildTaskToolResult]:
        return await asyncio.gather(
            *(
                cls.tool_pool[tool_name].invoke(request)
                for tool_name, request in requests
            )
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
            query_results=[
                ChildTaskQueryResult(
                    tool_name="rag",
                    original_query=request.query,
                    query_result=output.evidence,
                )
            ],
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
            return ChildTaskToolResult(
                query_results=[
                    ChildTaskQueryResult(
                        tool_name="sql",
                        original_query=request.query,
                    )
                ],
                tool_events=[event],
            )

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
                        query_results=[
                            ChildTaskQueryResult(
                                tool_name="sql",
                                original_query=request.query,
                                query_result=[evidence],
                                analysis=analysis,
                            )
                        ],
                        tool_events=[start_event, event],
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
            query_results=[
                ChildTaskQueryResult(
                    tool_name="sql",
                    original_query=request.query,
                )
            ],
            tool_events=[start_event, event],
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
        return ChildTaskToolResult(
            query_results=[
                ChildTaskQueryResult(
                    tool_name="website",
                    original_query=request.query,
                    query_result=evidence,
                )
            ],
            tool_events=events,
        )
