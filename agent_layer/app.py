# coding: utf-8

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from common.logger import get_logger, get_request_id
from agent_layer.classification.chain import QuestionClassifier
from agent_layer.config import Settings
from agent_layer.context import ContextResolver
from agent_layer.errors import AgentError
from agent_layer.schemas import (
    AskResult,
    Category,
    Citation,
    Evidence,
    GenerationOptions,
    Message,
    QuestionClassification,
    RouteDecision,
    SessionContext,
    StreamEvent,
    StreamEventType,
    ToolEvent,
)
from agent_layer.workflows.data_domain import DataDomainWorkflow
from agent_layer.workflows.general import GeneralWorkflow
from agent_layer.workflows.policy import PolicyWorkflow
from agent_layer.workflows.router import WorkflowRouter

logger = get_logger("agent.app")


@dataclass(frozen=True)
class ApplicationDependencies:
    settings: Settings
    context_resolver: ContextResolver
    classifier: QuestionClassifier
    router: WorkflowRouter
    general_workflow: GeneralWorkflow
    policy_workflow: PolicyWorkflow
    data_workflows: dict[Category, DataDomainWorkflow]


class TenderQAApplication:
    def __init__(self, dependencies: ApplicationDependencies):
        self.dependencies = dependencies

    async def stream(
        self,
        user_message: str,
        history_messages: list[Message] | None = None,
        session_context: SessionContext | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        started = time.perf_counter()
        history = history_messages or []
        context = session_context or SessionContext()
        options = generation_options or GenerationOptions()
        classification = None
        route = None
        classification_duration_ms = 0.0
        try:
            conversation = await self.dependencies.context_resolver.resolve(user_message, history, context)
            quick = self._quick_response(conversation.standalone_question)
            if quick:
                yield StreamEvent(
                    type=StreamEventType.ROUTE,
                    content="已识别为会话控制消息。",
                    data={"route": "greeting"},
                )
                async for event in self._answer_events(quick):
                    yield event
                yield self._final_event(
                    "greeting",
                    classification,
                    [],
                    [],
                    started,
                    conversation.context_model_calls,
                    0,
                    classification_duration_ms,
                )
                return

            if conversation.ambiguous:
                answer = conversation.clarification_question or "请明确当前问题所指的业务实体。"
                yield StreamEvent(
                    type=StreamEventType.ROUTE,
                    content="会话指代存在歧义，需要确认实体。",
                    data={"route": "clarify_entity"},
                )
                async for event in self._answer_events(answer):
                    yield event
                yield self._final_event(
                    "clarify_entity",
                    classification,
                    [],
                    [],
                    started,
                    conversation.context_model_calls,
                    0,
                    classification_duration_ms,
                )
                return

            classification_started = time.perf_counter()
            classification = await self.dependencies.classifier.classify(
                conversation.standalone_question,
                self.dependencies.context_resolver.context_summary(conversation),
            )
            classification_duration_ms = (time.perf_counter() - classification_started) * 1000
            route = self.dependencies.router.decide(classification)
            yield StreamEvent(
                type=StreamEventType.ROUTE,
                content=f"问题已路由到 {route.workflow} 工作流。",
                data={
                    "classification": classification.model_dump(mode="json"),
                    "route": route.model_dump(mode="json"),
                    "duration_ms": classification_duration_ms,
                },
            )
            yield StreamEvent(
                type=StreamEventType.REASONING_SUMMARY,
                content=route.reason,
                data={
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "requires_fresh_data": classification.requires_fresh_data,
                },
            )

            if route.action == "clarify":
                answer = self.dependencies.router.clarification_message(classification)
                async for event in self._answer_events(answer):
                    yield event
                yield self._final_event(
                    route.workflow,
                    classification,
                    [],
                    [],
                    started,
                    conversation.context_model_calls + 1,
                    0,
                    classification_duration_ms,
                )
                return

            if route.action == "general_answer":
                answer = await self.dependencies.general_workflow.run(
                    conversation.original_question,
                    self.dependencies.context_resolver.context_summary(conversation),
                )
                async for event in self._answer_events(answer):
                    yield event
                yield self._final_event(
                    route.workflow,
                    classification,
                    [],
                    [],
                    started,
                    conversation.context_model_calls + 2,
                    0,
                    classification_duration_ms,
                )
                return

            if options.include_progress:
                yield StreamEvent(
                    type=StreamEventType.PROGRESS,
                    content=f"正在执行{self._category_label(classification.category)}工作流。",
                )
            if classification.category == Category.POLICY:
                result = await self.dependencies.policy_workflow.run(
                    conversation.original_question,
                    conversation.standalone_question,
                    context,
                )
            else:
                workflow = self.dependencies.data_workflows[classification.category]
                result = await workflow.run(
                    conversation.standalone_question,
                    classification.entities,
                    classification.secondary_categories,
                    classification.requires_fresh_data,
                    context,
                )

            for tool_event in result.tool_events:
                if options.include_progress:
                    yield StreamEvent(
                        type=StreamEventType.PROGRESS,
                        content=tool_event.summary,
                        data=tool_event.model_dump(mode="json"),
                    )
            for item in result.evidence:
                yield StreamEvent(
                    type=StreamEventType.SOURCE,
                    content=item.title,
                    data={
                        "evidence_id": item.evidence_id,
                        "domain": item.domain.value,
                        "source_type": item.source_type.value,
                        "title": item.title,
                        "url": item.url,
                        "document_id": item.document_id,
                        "law_name": item.law_name,
                        "article_id": item.article_id,
                        "published_at": item.published_at.isoformat() if item.published_at else None,
                        "score": item.score,
                        "authority_level": item.authority_level,
                        "freshness_level": item.freshness_level,
                    },
                )
            async for event in self._answer_events(result.answer):
                yield event
            yield self._final_event(
                route.workflow,
                classification,
                result.citations,
                result.tool_events,
                started,
                conversation.context_model_calls + 1 + result.model_calls,
                len(result.evidence),
                classification_duration_ms,
                result.run_id,
                self._source_distribution(result.evidence),
            )
        except AgentError as exc:
            logger.warning("agent run failed | code=%s", exc.code, exc_info=True)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=exc.user_message,
                code=exc.code,
                error_id=get_request_id(),
            )
        except Exception:
            logger.exception("unexpected agent failure")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="系统已记录错误编号。",
                code="internal_error",
                error_id=get_request_id(),
            )

    async def ask(
        self,
        user_message: str,
        history_messages: list[Message] | None = None,
        session_context: SessionContext | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> AskResult:
        started = time.perf_counter()
        answer_parts = []
        route = "unknown"
        sources = []
        classification = None
        citations = []
        tool_events = []
        async for event in self.stream(user_message, history_messages, session_context, generation_options):
            if event.type == StreamEventType.ASSISTANT_DELTA:
                answer_parts.append(event.content)
            elif event.type == StreamEventType.ROUTE:
                route_data = event.data.get("route", route)
                if isinstance(route_data, dict):
                    route = route_data.get("workflow", route)
                elif isinstance(route_data, str):
                    route = route_data
                if event.data.get("classification"):
                    classification = QuestionClassification.model_validate(event.data["classification"])
            elif event.type == StreamEventType.SOURCE:
                sources.append(event.data)
            elif event.type == StreamEventType.FINAL:
                citations = [Citation.model_validate(item) for item in event.data.get("citations", [])]
                tool_events = [ToolEvent.model_validate(item) for item in event.data.get("tool_events", [])]
            elif event.type == StreamEventType.ERROR:
                answer_parts.append(event.content)
                route = "error"
        return AskResult(
            answer="".join(answer_parts),
            route=route,
            processing_time=time.perf_counter() - started,
            sources=sources,
            classification=classification,
            citations=citations,
            tool_events=tool_events,
        )

    async def _answer_events(self, answer: str) -> AsyncIterator[StreamEvent]:
        size = self.dependencies.settings.stream_chunk_size
        for index in range(0, len(answer), size):
            yield StreamEvent(
                type=StreamEventType.ASSISTANT_DELTA,
                content=answer[index : index + size],
            )

    def _quick_response(self, question: str) -> str | None:
        normalized = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", question.strip().lower())
        if not normalized:
            return "您好，请输入具体问题。"
        keyword_groups = [
            self.dependencies.settings.greeting_keywords,
            self.dependencies.settings.thanks_keywords,
            self.dependencies.settings.goodbye_keywords,
        ]
        for keywords in keyword_groups:
            for keyword in keywords:
                if normalized == keyword or (len(normalized) <= 8 and keyword in normalized):
                    return self.dependencies.settings.greeting_responses.get(
                        keyword,
                        "您好，请问有什么可以帮助您？",
                    )
        return None

    @staticmethod
    def _category_label(category: Category) -> str:
        return {
            Category.POLICY: "政策信息",
            Category.TENDER: "招标信息",
            Category.PUBLIC_OPINION: "舆情信息",
            Category.COMPANY: "企业信息",
            Category.PRICE: "价格信息",
            Category.PRODUCT: "商品信息",
            Category.OTHER: "通用回答",
        }[category]

    @staticmethod
    def _final_event(
        route: str,
        classification: QuestionClassification | None,
        citations: list[Citation],
        tool_events: list[ToolEvent],
        started: float,
        model_calls: int,
        evidence_count: int,
        classification_duration_ms: float,
        run_id: str | None = None,
        source_distribution: dict[str, int] | None = None,
    ) -> StreamEvent:
        retrieval_stages = {"policy_retrieve", "policy_internet", "local_domain_retrieve", "sql", "website"}
        generation_stages = {"policy_synthesis", "domain_synthesis"}
        retrieval_duration_ms = sum(
            event.duration_ms or 0
            for event in tool_events
            if event.stage in retrieval_stages
        )
        generation_duration_ms = sum(
            event.duration_ms or 0
            for event in tool_events
            if event.stage in generation_stages
        )
        tool_calls = sum(
            event.status in {"completed", "failed"}
            for event in tool_events
            if event.stage not in {"research_plan", "policy_parse", "policy_assessment", *generation_stages}
        )
        metrics = {
            "total_duration_ms": (time.perf_counter() - started) * 1000,
            "classification_duration_ms": classification_duration_ms,
            "retrieval_duration_ms": retrieval_duration_ms,
            "generation_duration_ms": generation_duration_ms,
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "evidence_count": evidence_count,
        }
        return StreamEvent(
            type=StreamEventType.FINAL,
            data={
                "route": route,
                "classification": classification.model_dump(mode="json") if classification else None,
                "citations": [citation.model_dump(mode="json") for citation in citations],
                "tool_events": [event.model_dump(mode="json") for event in tool_events],
                "metrics": metrics,
                "status": "completed",
                "run_id": run_id,
                "source_distribution": source_distribution or {},
            },
        )

    @staticmethod
    def _source_distribution(evidence: list[Evidence]) -> dict[str, int]:
        distribution = {}
        for item in evidence:
            key = item.source_type.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution
