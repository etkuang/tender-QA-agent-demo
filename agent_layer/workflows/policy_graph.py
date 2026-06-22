# coding: utf-8
# @Author: Wang Qingkang

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from common.logger import get_logger
from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.conversation.fallback_messages import NO_RESULTS_RESPONSE
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.config import Settings
from agent_layer.errors import CitationValidationError, GenerationError, raise_model_error
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.schemas import (
    Category,
    Evidence,
    PolicyQuery,
    RetrievalAssessment,
    SourceTier,
    ToolEvent,
    WebsiteQuery,
    WorkflowResult,
)
from agent_layer.workflows.common import (
    build_citations,
    citations_are_valid,
    ensure_source_section,
    format_evidence,
    rank_evidence,
)
from agent_layer.workflows.policy import POLICY_ANSWER_PROMPT, PolicyAssessmentChain, PolicyQueryParser
from agent_layer.workflows.self_rag import merge_evidence, next_retrieval_queries, select_evidence

logger = get_logger("agent.workflows.policy_graph")

ProgressCallback = Callable[[ToolEvent], Awaitable[None]]
RouteName = Literal["retrieve", "policy_internet", "assess", "synthesize"]


class PolicyGraphState(TypedDict, total=False):
    run_id: str
    original_question: str
    question: str
    history: str
    policy_query: PolicyQuery


class PolicyGraphWorkflow:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        query_parser: PolicyQueryParser,
        assessment: PolicyAssessmentChain,
        answer_model: BaseChatModel,
        adapter: EvidenceAdapter,
        settings: Settings,
        checkpoint_runtime: LangGraphCheckpointRuntime,
        internet_client: WebsiteSearchClient | None = None,
    ):
        self.retrieval = retrieval
        self.query_parser = query_parser
        self.assessment = assessment
        self.adapter = adapter
        self.settings = settings
        self.checkpoint_runtime = checkpoint_runtime
        self.internet_client = internet_client
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
        original_question: str,
        question: str,
        history: str,
        progress_callback: ProgressCallback | None = None,
    ) -> WorkflowResult:
        run_id = uuid.uuid4().hex
        if progress_callback is not None:
            self.progress_callbacks[run_id] = progress_callback
        initial_state = PolicyGraphState(
            run_id=run_id,
            original_question=original_question,
            question=question,
            history=history,
            evidence=[],
            retrieval_queries=[question],
            seen_queries=[question],
            round_index=0,
            model_calls=0,
            tool_events=[],
            internet_evidence_count=0,
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
        policy_query = await self.query_parser.parse(state["question"], state["history"])
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
        start_event = ToolEvent(
            stage="policy_retrieve",
            status="started",
            summary=f"Running policy retrieval round {round_index}.",
            details={"query": query, "round": round_index},
        )
        await self._emit(state["run_id"], start_event)
        retrieval_output = await self.retrieval.retrieve_policy(query, state.get("policy_query"))
        evidence = merge_evidence(state.get("evidence", []), retrieval_output.evidence)
        output_events = [event for event in retrieval_output.events if event.status != "started"]
        for event in output_events:
            await self._emit(state["run_id"], event)
        return {
            "retrieval_queries": remaining_queries,
            "round_index": round_index,
            "evidence": evidence,
            "tool_events": [*state.get("tool_events", []), start_event, *output_events],
        }

    async def _assess(self, state: PolicyGraphState) -> dict:
        started = time.perf_counter()
        evidence = rank_evidence(state.get("evidence", []))
        start_event = ToolEvent(
            stage="policy_assessment",
            status="started",
            summary="Checking whether current evidence is sufficient.",
            details={"round": state.get("round_index", 0), "evidence_count": len(evidence)},
        )
        await self._emit(state["run_id"], start_event)
        assessment = await self.assessment.assess(
            state["question"],
            format_evidence(
                evidence,
                self.settings.evidence_chunk_chars,
                max_total_chars=self.settings.evidence_context_chars,
            ),
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
            follow_up_queries = next_retrieval_queries(
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
        if assessment is None or assessment.sufficient:
            return "synthesize"
        if state.get("retrieval_queries") and state.get("round_index", 0) < self.settings.max_retrieval_rounds:
            return "retrieve"
        if assessment.need_official_web_search:
            return "policy_internet"
        return "synthesize"

    async def _policy_internet(self, state: PolicyGraphState) -> dict:
        assessment = state["assessment"]
        internet_evidence, internet_events = await self._search_internet(
            state["question"],
            assessment,
        )
        evidence = merge_evidence(state.get("evidence", []), internet_evidence)
        for event in internet_events:
            await self._emit(state["run_id"], event)
        return {
            "evidence": evidence,
            "internet_evidence_count": len(internet_evidence),
            "tool_events": [*state.get("tool_events", []), *internet_events],
        }

    def _route_after_internet(self, state: PolicyGraphState) -> RouteName:
        if state.get("internet_evidence_count", 0) > 0:
            return "assess"
        return "synthesize"

    async def _synthesize(self, state: PolicyGraphState) -> dict:
        evidence = select_evidence(state.get("evidence", []), state.get("assessment") or self._empty_assessment())
        if not evidence:
            result = WorkflowResult(
                answer=NO_RESULTS_RESPONSE,
                tool_events=state.get("tool_events", []),
                model_calls=state.get("model_calls", 0),
                run_id=state["run_id"],
            )
            return {"result": result}
        assessment = state.get("assessment") or self._empty_assessment()
        citations = build_citations(evidence)
        evidence_text = format_evidence(
            evidence,
            self.settings.evidence_chunk_chars,
            max_total_chars=self.settings.evidence_context_chars,
        )
        if not assessment.sufficient:
            missing_information = "；".join(assessment.missing_information)
            answer = f"Current evidence is insufficient for a reliable policy answer: {assessment.reason}"
            if missing_information:
                answer += f" Missing information: {missing_information}."
            answer = ensure_source_section(answer, citations)
            result = WorkflowResult(
                answer=answer,
                evidence=evidence,
                citations=citations,
                tool_events=state.get("tool_events", []),
                model_calls=state.get("model_calls", 0),
                run_id=state["run_id"],
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
            "original_question": state["original_question"],
            "question": state["question"],
            "history": state["history"],
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
    ) -> tuple[list[Evidence], list[ToolEvent]]:
        if not self.settings.policy_internet_enabled or self.internet_client is None:
            event = ToolEvent(
                stage="policy_internet",
                status="skipped",
                summary="Official policy internet adapter is not configured.",
                details={"missing_information": assessment.missing_information},
            )
            return [], [event]
        query_parts = assessment.follow_up_queries or assessment.missing_information
        query_text = "；".join(query_parts) or question
        website_query = WebsiteQuery(
            query=f"{question} {query_text}",
            category=Category.POLICY,
            keywords=assessment.missing_information,
        )
        started = time.perf_counter()
        try:
            results = await self.internet_client.search(
                website_query,
                self.settings.retrieval_batch_size,
            )
        except Exception:
            logger.warning("policy internet adapter failed", exc_info=True)
            event = ToolEvent(
                stage="policy_internet",
                status="failed",
                summary="Official policy internet adapter is temporarily unavailable.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return [], [event]
        official_results = [result for result in results if result.source_tier == SourceTier.OFFICIAL]
        evidence = [self.adapter.from_search_result(result, Category.POLICY) for result in official_results]
        event = ToolEvent(
            stage="policy_internet",
            status="completed",
            summary=f"Official policy internet search returned {len(evidence)} evidence items.",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"query": website_query.query},
        )
        return evidence, [event]

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