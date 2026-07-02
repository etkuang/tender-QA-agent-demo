# coding: utf-8
# @Author: Wang Qingkang

import time
import uuid
from collections.abc import Awaitable, Callable

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import (
    AgentError,
    CitationValidationError,
    SQLExecutionError,
    SQLValidationError,
)
from agent_layer.schemas import DependencyOutcome, ToolEvent, WorkflowResult
from agent_layer.workflows.common import (
    DomainProfile,
    build_citations,
    citations_are_valid,
    ensure_source_section,
    format_evidence,
)
from agent_layer.data_domain.analysis import (
    DataResultAnalyzer,
    build_context_evidence,
    build_sql_evidence,
)
from agent_layer.data_domain.chains import (
    DataAnswerChain,
    DataIntentClassifier,
    DataSQLGenerator,
    DataSQLRepairChain,
)
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.schemas import (
    DataQuestionType,
    SQLExecutionRequest,
    SQLRepairInput,
    SQLValidationResult,
)
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer

logger = get_logger("agent.data_domain.workflow")

ProgressCallback = Callable[[ToolEvent], Awaitable[None]]


class DataDomainWorkflow:
    def __init__(
        self,
        profile: DomainProfile,
        profiles: dict,
        context_retriever: DataContextRetriever,
        intent_classifier: DataIntentClassifier,
        sql_generator: DataSQLGenerator,
        sql_repair_chain: DataSQLRepairChain,
        sql_validator: SQLPolicyValidator,
        sql_gateway: SQLGateway | None,
        answer_chain: DataAnswerChain,
        web_supplementer: WebsiteSupplementer,
        analyzer: DataResultAnalyzer,
        settings: Settings,
    ):
        self.profile = profile
        self.profiles = profiles
        self.context_retriever = context_retriever
        self.intent_classifier = intent_classifier
        self.sql_generator = sql_generator
        self.sql_repair_chain = sql_repair_chain
        self.sql_validator = sql_validator
        self.sql_gateway = sql_gateway
        self.answer_chain = answer_chain
        self.web_supplementer = web_supplementer
        self.analyzer = analyzer
        self.settings = settings

    async def run(
        self,
        question: str,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        requires_fresh_data: bool,
        progress_callback: ProgressCallback | None = None,
    ) -> WorkflowResult:
        run_id = uuid.uuid4().hex
        events = []
        model_calls = 0

        await self._report(
            ToolEvent(stage="data_intent", status="started", summary="正在判断当前数据子任务的数据库问答类型和安全边界。"),
            progress_callback,
        )
        started = time.perf_counter()
        intent = await self.intent_classifier.classify(
            question,
            history,
            dependency_outcomes,
            self.profile,
            requires_fresh_data,
        )
        model_calls += 1
        intent_event = ToolEvent(
            stage="data_intent",
            status="completed",
            summary=f"已识别数据子任务类型：{intent.question_type.value}。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "question_type": intent.question_type.value,
                "requires_sql": intent.requires_sql,
                "requires_website": intent.requires_website,
                "requested_mode": intent.requested_mode.value,
            },
        )
        events.append(intent_event)
        await self._report(intent_event, progress_callback)

        if intent.question_type == DataQuestionType.UNSAFE_OR_UNSUPPORTED:
            reason = intent.unsafe_reason or intent.unsupported_reason or "该数据请求不在当前安全可执行范围内。"
            return WorkflowResult(
                answer=reason,
                tool_events=events,
                model_calls=model_calls,
                run_id=run_id,
                status="unsolved",
                unresolved_reason=reason,
            )
        if intent.clarification_question:
            return WorkflowResult(
                answer=intent.clarification_question,
                tool_events=events,
                model_calls=model_calls,
                run_id=run_id,
                status="unsolved",
                unresolved_reason=intent.clarification_question,
            )

        await self._report(
            ToolEvent(stage="data_context", status="started", summary="正在检索相关表、字段、关系、指标定义和示例 SQL。"),
            progress_callback,
        )
        started = time.perf_counter()
        context = self.context_retriever.retrieve(question, dependency_outcomes, intent, self.profile)
        context_event = ToolEvent(
            stage="data_context",
            status="completed",
            summary=f"已选取 {len(context.tables)} 张候选表、{len(context.metrics)} 个指标定义和 {len(context.examples)} 个示例。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "tables": sorted(context.table_names),
                "metrics": [metric.name for metric in context.metrics],
                "examples": [example.name for example in context.examples],
            },
        )
        events.append(context_event)
        await self._report(context_event, progress_callback)

        if intent.question_type == DataQuestionType.SCHEMA_DISCOVERY or not intent.requires_sql:
            return await self._answer_from_context(
                question,
                dependency_outcomes,
                intent,
                context,
                events,
                model_calls,
                run_id,
                progress_callback,
            )

        if self.sql_gateway is None:
            event = ToolEvent(
                stage="data_sql_execution",
                status="skipped",
                summary="结构化数据库查询尚未配置，数据子任务无法执行。",
            )
            events.append(event)
            await self._report(event, progress_callback)
            return WorkflowResult(
                answer=event.summary,
                tool_events=events,
                model_calls=model_calls,
                run_id=run_id,
                status="unsolved",
                unresolved_reason=event.summary,
            )

        sql_result, sql_statement, validation, sql_model_calls, sql_events = await self._generate_validate_execute_sql(
            question,
            dependency_outcomes,
            intent,
            context,
            run_id,
            progress_callback,
        )
        events.extend(sql_events)
        model_calls += sql_model_calls
        if sql_result is None:
            reason = self._sql_failure_reason(validation)
            return WorkflowResult(
                answer=reason,
                tool_events=events,
                model_calls=model_calls,
                run_id=run_id,
                status="unsolved",
                unresolved_reason=reason,
            )

        await self._report(
            ToolEvent(stage="data_analysis", status="started", summary="正在对 SQL 结果做确定性摘要和图表建议。"),
            progress_callback,
        )
        started = time.perf_counter()
        analysis = self.analyzer.analyze(sql_result, intent)
        analysis_event = ToolEvent(
            stage="data_analysis",
            status="completed",
            summary=f"已完成 {sql_result.row_count} 行 SQL 结果的确定性摘要。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "query_id": sql_result.query_id,
                "row_count": sql_result.row_count,
                "truncated": sql_result.truncated,
                "chart_suggestion": analysis.chart_suggestion,
            },
        )
        events.append(analysis_event)
        await self._report(analysis_event, progress_callback)

        evidence = [
            build_sql_evidence(
                sql_result,
                context,
                analysis,
                sql_statement,
                self.profile.category,
                f"{self.profile.display_name}结构化数据库查询",
            )
        ]
        web_evidence, web_events, web_errors = await self.web_supplementer.search(
            question,
            intent,
            context,
            self.profile,
            requires_fresh_data,
            progress_callback,
        )
        evidence.extend(web_evidence)
        events.extend(web_events)
        if web_errors and requires_fresh_data and not evidence:
            reason = web_errors[0].user_message
            return WorkflowResult(
                answer=reason,
                tool_events=events,
                model_calls=model_calls,
                run_id=run_id,
                status="unsolved",
                unresolved_reason=reason,
            )
        return await self._answer_with_evidence(
            question,
            dependency_outcomes,
            intent,
            context,
            sql_statement,
            sql_result,
            analysis,
            evidence,
            events,
            model_calls,
            run_id,
            progress_callback,
        )

    async def _answer_from_context(
        self,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent,
        context,
        events: list[ToolEvent],
        model_calls: int,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> WorkflowResult:
        evidence = [
            build_context_evidence(
                context,
                self.profile.category,
                f"{self.profile.display_name}数据上下文",
            )
        ]
        return await self._answer_with_evidence(
            question,
            dependency_outcomes,
            intent,
            context,
            "未执行 SQL；本回答仅基于数据上下文。",
            None,
            None,
            evidence,
            events,
            model_calls,
            run_id,
            progress_callback,
        )

    async def _generate_validate_execute_sql(
        self,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent,
        context,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ):
        events = []
        model_calls = 0
        await self._report(
            ToolEvent(stage="data_sql_generation", status="started", summary="正在根据语义上下文生成只读 SQL。"),
            progress_callback,
        )
        started = time.perf_counter()
        candidate = await self.sql_generator.generate(question, dependency_outcomes, intent, context)
        model_calls += 1
        event = ToolEvent(
            stage="data_sql_generation",
            status="completed",
            summary="已生成一条候选只读 SQL，准备进行安全校验。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "selected_tables": candidate.selected_tables,
                "selected_columns": candidate.selected_columns,
                "assumptions": candidate.assumptions,
            },
        )
        events.append(event)
        await self._report(event, progress_callback)

        validation = SQLValidationResult(valid=False, statement=candidate.statement)
        database_error = None
        for attempt in range(self.settings.sql_repair_attempts + 1):
            await self._report(
                ToolEvent(stage="data_sql_validation", status="started", summary="正在校验 SQL 语法、表字段、敏感字段、只读约束和 LIMIT。"),
                progress_callback,
            )
            started = time.perf_counter()
            validation = self.sql_validator.validate(candidate, context)
            status = "completed" if validation.valid else "failed"
            event = ToolEvent(
                stage="data_sql_validation",
                status=status,
                summary="SQL 校验通过。" if validation.valid else "SQL 未通过安全或结构校验。",
                duration_ms=(time.perf_counter() - started) * 1000,
                details={
                    "tables": validation.tables,
                    "columns": validation.columns,
                    "issues": [issue.model_dump() for issue in validation.issues],
                    "attempt": attempt,
                },
            )
            events.append(event)
            await self._report(event, progress_callback)
            if validation.valid:
                await self._report(
                    ToolEvent(stage="data_sql_execution", status="started", summary="正在使用只读数据库连接执行已校验 SQL。"),
                    progress_callback,
                )
                started = time.perf_counter()
                try:
                    result = await self.sql_gateway.execute(
                        SQLExecutionRequest(
                            statement=validation.statement,
                            parameters=candidate.parameters,
                            timeout_seconds=self.settings.sql_statement_timeout_seconds,
                            max_rows=self.settings.sql_max_rows,
                            domain=self.profile.category,
                            question_type=intent.question_type,
                            selected_tables=validation.tables,
                            selected_columns=validation.columns,
                            request_id=run_id,
                        )
                    )
                except SQLExecutionError as exc:
                    database_error = exc.user_message
                    event = ToolEvent(
                        stage="data_sql_execution",
                        status="failed",
                        summary=exc.user_message,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        details={"attempt": attempt},
                    )
                    events.append(event)
                    await self._report(event, progress_callback)
                else:
                    event = ToolEvent(
                        stage="data_sql_execution",
                        status="completed",
                        summary=f"只读 SQL 执行完成，返回 {result.row_count} 行。",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        details={
                            "query_id": result.query_id,
                            "row_count": result.row_count,
                            "truncated": result.truncated,
                        },
                    )
                    events.append(event)
                    await self._report(event, progress_callback)
                    return result, validation.statement, validation, model_calls, events

            if attempt >= self.settings.sql_repair_attempts:
                break
            await self._report(
                ToolEvent(stage="data_sql_repair", status="started", summary="正在根据校验或执行错误进行受限 SQL 修复。"),
                progress_callback,
            )
            started = time.perf_counter()
            candidate = await self.sql_repair_chain.repair(
                SQLRepairInput(
                    original_question=question,
                    previous_sql=candidate.statement,
                    validation_issues=validation.issues,
                    database_error=database_error,
                    attempt_index=attempt + 1,
                ),
                intent,
                context,
            )
            model_calls += 1
            event = ToolEvent(
                stage="data_sql_repair",
                status="completed",
                summary="已生成修复后的候选 SQL。",
                duration_ms=(time.perf_counter() - started) * 1000,
                details={"attempt": attempt + 1},
            )
            events.append(event)
            await self._report(event, progress_callback)
        return None, candidate.statement, validation, model_calls, events

    async def _answer_with_evidence(
        self,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent,
        context,
        sql_statement: str,
        sql_result,
        analysis,
        evidence,
        events: list[ToolEvent],
        model_calls: int,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> WorkflowResult:
        await self._report(
            ToolEvent(stage="data_synthesis", status="started", summary="正在基于数据库证据和补充证据生成数据子任务回答。"),
            progress_callback,
        )
        started = time.perf_counter()
        citations = build_citations(evidence)
        sql_for_answer = sql_statement if self.settings.sql_show_generated_sql else "配置为不展示 SQL。"
        web_evidence = format_evidence(
            [item for item in evidence if item.source_type.value == "website"],
            self.settings.evidence_chunk_chars,
            max_total_chars=self.settings.evidence_context_chars,
        )
        answer = await self.answer_chain.answer(
            self.profile,
            question,
            dependency_outcomes,
            intent,
            context,
            sql_for_answer,
            sql_result,
            analysis,
            web_evidence,
            "",
        )
        model_calls += 1
        if not citations_are_valid(answer, len(citations)):
            answer = await self.answer_chain.answer(
                self.profile,
                question,
                dependency_outcomes,
                intent,
                context,
                sql_for_answer,
                sql_result,
                analysis,
                web_evidence,
                "上一版引用缺失或编号越界。请仅使用现有 [1] 到 [N] 编号重写。",
            )
            model_calls += 1
        if not citations_are_valid(answer, len(citations)):
            raise CitationValidationError
        answer = ensure_source_section(answer, citations)
        event = ToolEvent(
            stage="data_synthesis",
            status="completed",
            summary=f"{self.profile.display_name}数据子任务回答完成。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"citation_count": len(citations)},
        )
        events.append(event)
        await self._report(event, progress_callback)
        return WorkflowResult(
            answer=answer,
            evidence=evidence,
            citations=citations,
            tool_events=events,
            model_calls=model_calls,
            run_id=run_id,
        )

    @staticmethod
    def _sql_failure_reason(validation: SQLValidationResult) -> str:
        if validation.issues:
            return "SQL 生成未通过安全或结构校验：" + "；".join(issue.message for issue in validation.issues)
        return "SQL 生成、校验或执行失败，系统未返回未经核验的数据结论。"

    @staticmethod
    async def _report(event: ToolEvent, progress_callback: ProgressCallback | None) -> None:
        if progress_callback is not None:
            await progress_callback(event)