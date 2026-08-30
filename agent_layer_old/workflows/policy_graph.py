# coding: utf-8
# @Author: Wang Qingkang

import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from common.logger import get_logger
from agent_layer_old.conversation.fallback_messages import NO_RESULTS_RESPONSE
from agent_layer_old.checkpoint import LangGraphCheckpointRuntime
from agent_layer_old.config import Settings
from agent_layer_old.errors import CitationValidationError, GenerationError, raise_model_error
from agent_layer_old.schemas import (
    Category,
    DependencyOutcome,
    Evidence,
    PolicyQuery,
    RetrievalAssessment,
    SourceTier,
    ToolEvent,
    WorkflowResult,
)
from agent_layer_old.workflows.common import (
    build_citations,
    citations_are_valid,
    ensure_source_section,
    format_dependency_outcomes,
    prepare_evidence_context,
    rank_evidence,
)
from agent_layer_old.workflows.tools import ChildTaskToolRegistry, ChildTaskToolRequest
from agent_layer_old.workflows.policy import POLICY_ANSWER_PROMPT, PolicyAssessmentChain, PolicyQueryParser
from agent_layer_old.workflows.self_rag import PolicySelfRAGPlugin

logger = get_logger("agent.workflows.policy_graph")

RouteName = Literal["retrieve", "policy_internet", "assess", "synthesize"]


class PolicyGraphState(TypedDict, total=False):
    run_id: str
    question: str
    history: str
    dependency_outcomes: list[DependencyOutcome]
    requires_fresh_data: bool
    policy_query: PolicyQuery
    evidence: list[Evidence]
    assessment: RetrievalAssessment
    retrieval_queries: list[str]
    seen_queries: list[str]
    round_index: int
    model_calls: int
    tool_events: list[ToolEvent]
    result: WorkflowResult
    internet_evidence_count: int
    internet_searched: bool


class PolicyWorkflow:
    def __init__(
        self,
        query_parser: PolicyQueryParser,
        assessment: PolicyAssessmentChain,
        answer_model: BaseChatModel,
        settings: Settings,
        checkpoint_runtime: LangGraphCheckpointRuntime,
        self_rag: PolicySelfRAGPlugin,
    ):
        self.query_parser = query_parser
        self.assessment = assessment
        self.settings = settings
        self.checkpoint_runtime = checkpoint_runtime
        self.self_rag = self_rag
        self.answer_chain = POLICY_ANSWER_PROMPT | answer_model | StrOutputParser()
        self.progress_callbacks = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(PolicyGraphState)
        builder.add_node("parse_policy", self._parse_policy)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess", self._assess)
        builder.add_node("policy_internet", self._policy_internet)
        builder.add_node("synthesize", self._synthesize)
        builder.set_entry_point("parse_policy")
        builder.add_edge("parse_policy", "retrieve")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {
                "retrieve": "retrieve",
                "policy_internet": "policy_internet",
                "synthesize": "synthesize",
            },
        )
        builder.add_conditional_edges(
            "policy_internet",
            self._route_after_internet,
            {
                "assess": "assess",
                "synthesize": "synthesize",
            },
        )
        builder.add_edge("synthesize", END)
        return builder.compile(checkpointer=self.checkpoint_runtime.saver)

    async def run(
        self,
        run_id: str,
        question: str,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        requires_fresh_data: bool,
        progress_callback: Callable[[ToolEvent], Awaitable[None]],
    ) -> WorkflowResult:
        self.progress_callbacks[run_id] = progress_callback
        initial_state = PolicyGraphState(
            run_id=run_id,
            question=question,
            history=history,
            dependency_outcomes=dependency_outcomes,
            requires_fresh_data=requires_fresh_data,
            evidence=[],
            retrieval_queries=[question],
            seen_queries=[question],
            round_index=0,
            model_calls=0,
            tool_events=[],
            internet_evidence_count=0,
            internet_searched=False,
        )
        try:
            state = await self.graph.ainvoke(initial_state, self.checkpoint_runtime.config(run_id))
        finally:
            self.progress_callbacks.pop(run_id, None)
        result = state["result"]
        if isinstance(result, WorkflowResult):
            return result
        return WorkflowResult.model_validate(result)

    async def load(self, run_id: str) -> dict[str, Any] | None:
        checkpoint = await self.checkpoint_runtime.load(run_id)
        if checkpoint is None:
            return None
        return checkpoint.checkpoint

    async def _parse_policy(self, state: PolicyGraphState) -> dict:
        started = time.perf_counter()
        start_event = ToolEvent(
            stage="policy_parse",
            status="started",
            summary="Parsing policy query conditions.",
        )
        await self._emit(state["run_id"], start_event)
        policy_query = await self.query_parser.parse(
            state["question"],
            state["history"],
            state["dependency_outcomes"],
        )
        event = ToolEvent(
            stage="policy_parse",
            status="completed",
            summary="Policy query conditions parsed.",
            duration_ms=(time.perf_counter() - started) * 1000,
            details=policy_query.model_dump(mode="json"),
        )
        await self._emit(state["run_id"], event)
        return {
            "policy_query": policy_query,
            "model_calls": state.get("model_calls", 0) + 1,
            "tool_events": [*state.get("tool_events", []), start_event, event],
        }

    async def _retrieve(self, state: PolicyGraphState) -> dict:
        queries = state.get("retrieval_queries", [])
        if not queries:
            return {}
        query = queries[0]
        remaining_queries = queries[1:]
        round_index = state.get("round_index", 0) + 1
        tool_result = await ChildTaskToolRegistry.invoke(
            "rag",
            ChildTaskToolRequest(
                category=Category.POLICY,
                query=query,
                run_id=state["run_id"],
                progress_callback=self.progress_callbacks[state["run_id"]],
                policy_query=state.get("policy_query"),
            ),
        )
        evidence = self.self_rag.merge_evidence(
            state.get("evidence", []),
            tool_result.collect_evidence(),
        )
        return {
            "retrieval_queries": remaining_queries,
            "round_index": round_index,
            "evidence": evidence,
            "tool_events": [
                *state.get("tool_events", []),
                *tool_result.tool_events,
            ],
        }

    async def _assess(self, state: PolicyGraphState) -> dict:
        started = time.perf_counter()
        evidence, evidence_text = prepare_evidence_context(
            rank_evidence(state.get("evidence", [])),
            self.settings.evidence_chunk_chars,
            self.settings.evidence_context_chars,
        )
        start_event = ToolEvent(
            stage="policy_assessment",
            status="started",
            summary="Checking whether current evidence is sufficient.",
            details={"round": state.get("round_index", 0), "evidence_count": len(evidence)},
        )
        await self._emit(state["run_id"], start_event)
        assessment = await self.assessment.assess(
            state["question"],
            state["dependency_outcomes"],
            evidence_text,
            len(evidence),
        )
        event = ToolEvent(
            stage="policy_assessment",
            status="completed",
            summary="Evidence sufficiency check completed.",
            duration_ms=(time.perf_counter() - started) * 1000,
            details=assessment.model_dump(),
        )
        await self._emit(state["run_id"], event)
        updates = {
            "assessment": assessment,
            "model_calls": state.get("model_calls", 0) + (1 if evidence else 0),
            "tool_events": [*state.get("tool_events", []), start_event, event],
        }
        if not assessment.sufficient and assessment.need_more_local_retrieval:
            seen_queries = set(state.get("seen_queries", []))
            follow_up_queries = self.self_rag.next_retrieval_queries(
                assessment,
                seen_queries,
                self.settings.max_follow_up_queries,
            )
            if follow_up_queries:
                updates["retrieval_queries"] = [*state.get("retrieval_queries", []), *follow_up_queries]
                updates["seen_queries"] = sorted(seen_queries)
        return updates

    def _route_after_assessment(self, state: PolicyGraphState) -> RouteName:
        assessment = state.get("assessment")
        if assessment is None:
            return "synthesize"
        internet_needed = state.get("requires_fresh_data", False) or assessment.need_official_web_search
        internet_available = not state.get("internet_searched", False)
        if assessment.sufficient:
            if internet_needed and internet_available:
                return "policy_internet"
            return "synthesize"
        if state.get("retrieval_queries") and state.get("round_index", 0) < self.settings.max_retrieval_rounds:
            return "retrieve"
        if internet_needed and internet_available:
            return "policy_internet"
        return "synthesize"

    async def _policy_internet(self, state: PolicyGraphState) -> dict:
        assessment = state["assessment"]
        internet_evidence, internet_events = await self._search_internet(
            state["question"],
            assessment,
            state["run_id"],
        )
        evidence = self.self_rag.merge_evidence(
            state.get("evidence", []),
            internet_evidence,
        )
        return {
            "evidence": evidence,
            "internet_evidence_count": len(internet_evidence),
            "internet_searched": True,
            "tool_events": [*state.get("tool_events", []), *internet_events],
        }

    def _route_after_internet(self, state: PolicyGraphState) -> RouteName:
        if state.get("internet_evidence_count", 0) > 0:
            return "assess"
        return "synthesize"

    async def _synthesize(self, state: PolicyGraphState) -> dict:
        evidence, evidence_text = prepare_evidence_context(
            self.self_rag.select_evidence(
                state.get("evidence", []),
                state.get("assessment") or self._empty_assessment(),
            ),
            self.settings.evidence_chunk_chars,
            self.settings.evidence_context_chars,
        )
        if not evidence:
            result = WorkflowResult(
                answer=None,
                tool_events=state.get("tool_events", []),
                model_calls=state.get("model_calls", 0),
                run_id=state["run_id"],
                status="unsolved",
                unresolved_reason=NO_RESULTS_RESPONSE,
            )
            return {"result": result}
        assessment = state.get("assessment") or self._empty_assessment()
        citations = build_citations(evidence)
        if not assessment.sufficient:
            missing_information = "；".join(assessment.missing_information)
            reason = f"当前政策证据不足，无法可靠回答：{assessment.reason}"
            if missing_information:
                reason += f" 缺少信息：{missing_information}。"
            result = WorkflowResult(
                answer=None,
                evidence=evidence,
                citations=citations,
                tool_events=state.get("tool_events", []),
                model_calls=state.get("model_calls", 0),
                run_id=state["run_id"],
                status="unsolved",
                unresolved_reason=reason,
            )
            return {"result": result}

        started = time.perf_counter()
        start_event = ToolEvent(
            stage="policy_synthesis",
            status="started",
            summary="Synthesizing the final policy answer.",
        )
        await self._emit(state["run_id"], start_event)
        answer_input = {
            "question": state["question"],
            "history": state["history"],
            "dependency_outcomes": format_dependency_outcomes(state["dependency_outcomes"]),
            "assessment": assessment.model_dump_json(),
            "evidence": evidence_text,
            "citation_feedback": "",
        }
        answer = await self._generate_answer(answer_input)
        model_calls = state.get("model_calls", 0) + 1
        if not citations_are_valid(answer, len(citations)):
            answer_input["citation_feedback"] = "Previous citation markers were missing or out of range. Rewrite using only available citation numbers."
            answer = await self._generate_answer(answer_input)
            model_calls += 1
        if not citations_are_valid(answer, len(citations)):
            raise CitationValidationError
        answer = ensure_source_section(answer, citations)
        event = ToolEvent(
            stage="policy_synthesis",
            status="completed",
            summary="Policy answer synthesis completed.",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"citation_count": len(citations)},
        )
        await self._emit(state["run_id"], event)
        result = WorkflowResult(
            answer=answer,
            evidence=evidence,
            citations=citations,
            tool_events=[*state.get("tool_events", []), start_event, event],
            model_calls=model_calls,
            run_id=state["run_id"],
        )
        return {
            "result": result,
            "model_calls": model_calls,
            "tool_events": [*state.get("tool_events", []), start_event, event],
        }

    async def _search_internet(
        self,
        question: str,
        assessment: RetrievalAssessment,
        run_id: str,
    ) -> tuple[list[Evidence], list[ToolEvent]]:
        if not self.settings.policy_internet_enabled:
            event = ToolEvent(
                stage="policy_internet",
                status="skipped",
                summary="未启用政策官方网站检索。",
                details={"missing_information": assessment.missing_information},
            )
            await self._emit(run_id, event)
            return [], [event]
        query_parts = assessment.follow_up_queries or assessment.missing_information
        query_text = "；".join(query_parts) or question
        tool_result = await ChildTaskToolRegistry.invoke(
            "website",
            ChildTaskToolRequest(
                category=Category.POLICY,
                query=f"{question} {query_text}",
                run_id=run_id,
                progress_callback=self.progress_callbacks[run_id],
                keywords=assessment.missing_information,
                required_source_tier=SourceTier.OFFICIAL,
            ),
        )
        return tool_result.collect_evidence(), tool_result.tool_events

    async def _generate_answer(self, answer_input: dict) -> str:
        try:
            return await self.answer_chain.ainvoke(answer_input)
        except Exception as exc:
            logger.exception("policy answer generation failed")
            raise_model_error(exc, GenerationError)

    async def _emit(self, run_id: str, event: ToolEvent) -> None:
        callback = self.progress_callbacks.get(run_id)
        if callback is not None:
            await callback(event)

    @staticmethod
    def _empty_assessment() -> RetrievalAssessment:
        return RetrievalAssessment(sufficient=False, reason="Evidence assessment has not completed.")