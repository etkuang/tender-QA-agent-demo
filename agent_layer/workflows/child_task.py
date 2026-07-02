# coding: utf-8
# @Author: Wang Qingkang

import time
import uuid
from collections.abc import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import CitationValidationError, SQLExecutionError, raise_model_error
from agent_layer.schemas import Category, DataResult, DependencyOutcome, Evidence, SourceType, ToolEvent, WorkflowResult
from agent_layer.workflows.common import (
    ChildTaskWorkflowProfile,
    ToolName,
    build_citations,
    citations_are_valid,
    ensure_source_section,
    format_dependency_outcomes,
    format_evidence,
    format_workflow_profile,
)
from agent_layer.data_domain.analysis import DataResultAnalyzer, build_context_evidence, build_sql_evidence
from agent_layer.data_domain.chains import (
    DataSQLGenerator,
    DataSQLRepairChain,
    format_analysis,
    format_data_context,
    format_sql_result,
)
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.schemas import (
    DataAnalysisSummary,
    DataContextBundle,
    DataIntentDecision,
    DataQuestionType,
    SQLExecutionRequest,
    SQLRepairInput,
    SQLValidationResult,
)
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer

logger = get_logger("agent.child_workflow")

ProgressCallback = Callable[[ToolEvent], Awaitable[None]]

CHILD_INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You classify a single child task. Tool availability and fallback order are controlled by the workflow profile, not by your output.
Return only JSON matching DataIntentDecision. Use unsafe_or_unsupported for write operations, permission bypass, sensitive bulk extraction, or unsupported requests.
History is intent context only, not evidence. If a business term is ambiguous enough to change the answer, provide clarification_question.""",
        ),
        (
            "human",
            "workflow profile:\n{profile}\n\ncategory: {category}\n\nchild task:\n{question}\n\n"
            "dependency outcomes:\n{dependency_outcomes}\n\nhistory:\n{history}\n\nrequires fresh data: {requires_fresh_data}",
        ),
    ]
)

CHILD_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the child task using only the provided evidence and deterministic SQL analysis.
Do not fabricate database rows, website facts, policy clauses, company identities, product parameters, or metric definitions.
If sources disagree, state the disagreement truthfully, for example: SQL shows X, while website evidence shows Y.
Use [1], [2] citation markers for key facts when evidence exists. If the model-only fallback is used, say that no external evidence was available and do not cite.""",
        ),
        (
            "human",
            "workflow profile:\n{profile}\n\ncategory: {category}\n\nquestion:\n{question}\n\nintent:\n{intent}\n\n"
            "dependency outcomes:\n{dependency_outcomes}\n\nhistory:\n{history}\n\nexecuted tiers:\n{executed_tiers}\n\n"
            "data context:\n{context}\n\nSQL:\n{sql}\n\nSQL result:\n{sql_result}\n\nanalysis:\n{analysis}\n\n"
            "evidence:\n{evidence}\n\ncitation feedback:\n{citation_feedback}",
        ),
    ]
)


class ChildIntentClassifier:
    def __init__(self, model: BaseChatModel, settings: Settings):
        structured = model.with_structured_output(DataIntentDecision, method="json_mode")
        self.chain = CHILD_INTENT_PROMPT | structured
        self.settings = settings

    async def classify(
        self,
        category: Category,
        profile: ChildTaskWorkflowProfile,
        question: str,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        requires_fresh_data: bool,
    ) -> DataIntentDecision:
        try:
            return await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "profile": format_workflow_profile(profile),
                    "category": category.value,
                    "question": question,
                    "history": history,
                    "dependency_outcomes": format_dependency_outcomes(dependency_outcomes),
                    "requires_fresh_data": requires_fresh_data,
                }
            )
        except Exception as exc:
            logger.exception("child task intent classification failed | category=%s", category.value)
            raise_model_error(exc, SQLExecutionError)


class ChildAnswerChain:
    def __init__(self, model: BaseChatModel):
        self.chain = CHILD_ANSWER_PROMPT | model | StrOutputParser()

    async def answer(self, values: dict) -> str:
        return await self.chain.ainvoke(values)


class GeneralChildTaskWorkflow:
    def __init__(
        self,
        category: Category,
        profile: ChildTaskWorkflowProfile,
        intent_classifier: ChildIntentClassifier,
        sql_generator: DataSQLGenerator,
        sql_repair_chain: DataSQLRepairChain,
        sql_validator: SQLPolicyValidator,
        sql_gateway: SQLGateway | None,
        context_retriever: DataContextRetriever,
        web_supplementer: WebsiteSupplementer,
        analyzer: DataResultAnalyzer,
        answer_chain: ChildAnswerChain,
        settings: Settings,
    ):
        self.category = category
        self.profile = profile
        self.intent_classifier = intent_classifier
        self.sql_generator = sql_generator
        self.sql_repair_chain = sql_repair_chain
        self.sql_validator = sql_validator
        self.sql_gateway = sql_gateway
        self.context_retriever = context_retriever
        self.web_supplementer = web_supplementer
        self.analyzer = analyzer
        self.answer_chain = answer_chain
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
        intent = await self.intent_classifier.classify(
            self.category,
            self.profile,
            question,
            history,
            dependency_outcomes,
            requires_fresh_data,
        )
        model_calls += 1

        if intent.question_type == DataQuestionType.UNSAFE_OR_UNSUPPORTED:
            reason = intent.unsafe_reason or intent.unsupported_reason or "This child task is outside the supported safe scope."
            return WorkflowResult(answer=reason, tool_events=events, model_calls=model_calls, run_id=run_id, status="unsolved", unresolved_reason=reason)
        if intent.clarification_question:
            return WorkflowResult(answer=intent.clarification_question, tool_events=events, model_calls=model_calls, run_id=run_id, status="unsolved", unresolved_reason=intent.clarification_question)

        evidence = []
        executed_tiers = []
        context = None
        sql_statement = "No SQL was executed."
        sql_result = None
        analysis = None
        used_model_only = False

        for tier in self.profile.tool_preference:
            tier_evidence, tier_events, tier_context, tier_sql, tier_sql_result, tier_analysis, tier_model_only = await self._run_tier(
                tier,
                question,
                dependency_outcomes,
                intent,
                run_id,
                progress_callback,
            )
            evidence.extend(tier_evidence)
            events.extend(tier_events)
            context = tier_context or context
            sql_statement = tier_sql or sql_statement
            sql_result = tier_sql_result or sql_result
            analysis = tier_analysis or analysis
            used_model_only = tier_model_only
            eligible = used_model_only or self._has_eligible_evidence(tier_evidence, requires_fresh_data)
            executed_tiers.append({"tools": tier, "evidence_count": len(tier_evidence), "eligible": eligible})
            if eligible:
                break

        if not evidence and not used_model_only:
            reason = "No eligible evidence was found from the configured workflow tools."
            return WorkflowResult(answer=reason, tool_events=events, model_calls=model_calls, run_id=run_id, status="unsolved", unresolved_reason=reason)

        citations = build_citations(evidence)
        answer = await self._answer(
            question,
            history,
            dependency_outcomes,
            intent,
            executed_tiers,
            context,
            sql_statement,
            sql_result,
            analysis,
            evidence,
            "",
        )
        model_calls += 1
        if not citations_are_valid(answer, len(citations)):
            answer = await self._answer(
                question,
                history,
                dependency_outcomes,
                intent,
                executed_tiers,
                context,
                sql_statement,
                sql_result,
                analysis,
                evidence,
                "Previous citation markers were missing or out of range. Rewrite using only available citation numbers.",
            )
            model_calls += 1
        if not citations_are_valid(answer, len(citations)):
            raise CitationValidationError
        answer = ensure_source_section(answer, citations)
        return WorkflowResult(answer=answer, evidence=evidence, citations=citations, tool_events=events, model_calls=model_calls, run_id=run_id)

    async def _run_tier(
        self,
        tier: list[ToolName],
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[Evidence], list[ToolEvent], DataContextBundle | None, str | None, DataResult | None, DataAnalysisSummary | None, bool]:
        evidence = []
        events = []
        context = None
        sql_statement = None
        sql_result = None
        analysis = None
        used_model_only = False
        for tool in tier:
            if tool == "sql":
                tool_evidence, tool_events, context, sql_statement, sql_result, analysis = await self._run_sql(
                    question,
                    dependency_outcomes,
                    intent,
                    run_id,
                    progress_callback,
                )
                evidence.extend(tool_evidence)
                events.extend(tool_events)
            elif tool == "website":
                tool_evidence, tool_events, _ = await self.web_supplementer.search(
                    question,
                    intent,
                    self.category,
                    progress_callback,
                )
                evidence.extend(tool_evidence)
                events.extend(tool_events)
            elif tool == "model_only":
                used_model_only = True
        return evidence, events, context, sql_statement, sql_result, analysis, used_model_only

    async def _run_sql(
        self,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[Evidence], list[ToolEvent], DataContextBundle, str | None, DataResult | None, DataAnalysisSummary | None]:
        events = []
        context = self.context_retriever.retrieve(question, dependency_outcomes, intent, self.category)
        if intent.question_type == DataQuestionType.SCHEMA_DISCOVERY:
            return [build_context_evidence(context, self.category, f"{self.category.value} SQL context")], events, context, "No SQL was executed.", None, None
        if self.sql_gateway is None:
            event = ToolEvent(stage="sql_execution", status="skipped", summary="Read-only SQL gateway is not configured.")
            await self._report(event, progress_callback)
            return [], [event], context, None, None, None

        sql_result, sql_statement, validation, sql_events = await self._generate_validate_execute_sql(
            question,
            dependency_outcomes,
            intent,
            context,
            run_id,
            progress_callback,
        )
        events.extend(sql_events)
        if sql_result is None:
            return [], events, context, sql_statement, None, None
        analysis = self.analyzer.analyze(sql_result, intent)
        evidence = [build_sql_evidence(sql_result, context, analysis, sql_statement, self.category, f"{self.category.value} SQL query result")]
        return evidence, events, context, sql_statement, sql_result, analysis

    async def _generate_validate_execute_sql(
        self,
        question: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        context: DataContextBundle,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> tuple[DataResult | None, str, SQLValidationResult, list[ToolEvent]]:
        events = []
        candidate = await self.sql_generator.generate(question, dependency_outcomes, intent, context)
        validation = SQLValidationResult(valid=False, statement=candidate.statement)
        database_error = None
        for attempt in range(self.settings.sql_repair_attempts + 1):
            validation = self.sql_validator.validate(candidate, context)
            if validation.valid:
                try:
                    result = await self.sql_gateway.execute(
                        SQLExecutionRequest(
                            statement=validation.statement,
                            parameters=candidate.parameters,
                            timeout_seconds=self.settings.sql_statement_timeout_seconds,
                            max_rows=self.settings.sql_max_rows,
                            domain=self.category,
                            question_type=intent.question_type,
                            selected_tables=validation.tables,
                            selected_columns=validation.columns,
                            request_id=run_id,
                        )
                    )
                except SQLExecutionError as exc:
                    database_error = exc.user_message
                    events.append(ToolEvent(stage="sql_execution", status="failed", summary=exc.user_message))
                else:
                    events.append(ToolEvent(stage="sql_execution", status="completed", summary=f"Read-only SQL execution completed with {result.row_count} rows."))
                    return result, validation.statement, validation, events
            if attempt >= self.settings.sql_repair_attempts:
                break
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
        return None, candidate.statement, validation, events

    async def _answer(
        self,
        question: str,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        intent: DataIntentDecision,
        executed_tiers: list[dict],
        context: DataContextBundle | None,
        sql_statement: str,
        sql_result: DataResult | None,
        analysis: DataAnalysisSummary | None,
        evidence: list[Evidence],
        citation_feedback: str,
    ) -> str:
        evidence_text = format_evidence(evidence, self.settings.evidence_chunk_chars, max_total_chars=self.settings.evidence_context_chars) if evidence else "No external evidence."
        return await self.answer_chain.answer(
            {
                "profile": format_workflow_profile(self.profile),
                "category": self.category.value,
                "question": question,
                "intent": intent.model_dump_json(),
                "dependency_outcomes": format_dependency_outcomes(dependency_outcomes),
                "history": history,
                "executed_tiers": executed_tiers,
                "context": format_data_context(context) if context else "No SQL context.",
                "sql": sql_statement,
                "sql_result": format_sql_result(sql_result),
                "analysis": format_analysis(analysis),
                "evidence": evidence_text,
                "citation_feedback": citation_feedback,
            }
        )

    @staticmethod
    def _has_eligible_evidence(evidence: list[Evidence], requires_fresh_data: bool) -> bool:
        for item in evidence:
            if item.source_type == SourceType.SQL and item.metadata.get("row_count") == 0:
                continue
            if requires_fresh_data and item.freshness_level <= 0:
                continue
            return True
        return False

    @staticmethod
    async def _report(event: ToolEvent, progress_callback: ProgressCallback | None) -> None:
        if progress_callback is not None:
            await progress_callback(event)
