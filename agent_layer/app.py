# coding: utf-8

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from common.api_contracts.agent_api import Message
from common.logger import get_logger
from agent_layer.classification.chain import QuestionClassifier
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.config import Settings
from agent_layer.conversation.quick_responses import EMPTY_QUICK_RESPONSE, QUICK_RESPONSES
from agent_layer.errors import AgentError, ClassificationError
from agent_layer.schemas import (
    Category,
    ChildTask,
    Evidence,
    RoutePlan,
    StreamEvent,
    StreamEventType,
    WorkflowResult,
)
from agent_layer.workflows.data_domain import DataDomainWorkflow
from agent_layer.workflows.general import CompositeAnswerWorkflow, GeneralWorkflow
from agent_layer.workflows.policy_graph import PolicyGraphWorkflow
from agent_layer.workflows.router import WorkflowRouter

logger = get_logger("agent.app")


@dataclass(frozen=True)
class ApplicationDependencies:
    settings: Settings
    classifier: QuestionClassifier
    router: WorkflowRouter
    general_workflow: GeneralWorkflow
    composite_workflow: CompositeAnswerWorkflow
    policy_workflow: PolicyGraphWorkflow
    data_workflows: dict[Category, DataDomainWorkflow]
    checkpoint_runtime: LangGraphCheckpointRuntime


class TenderQAApplication:
    def __init__(self, dependencies: ApplicationDependencies):
        self.dependencies = dependencies

    async def aclose(self) -> None:
        await self.dependencies.checkpoint_runtime.close()

    async def stream(
        self,
        user_message: str,
        history_messages: list[Message] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        history = history_messages or []
        try:
            quick = self._quick_response(user_message)
            if quick:
                yield StreamEvent(
                    type=StreamEventType.ROUTE,
                    content="已识别为会话控制消息。",
                    data={"route": quick["kind"]},
                )
                async for event in self._answer_events(quick["content"]):
                    yield event
                return

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="正在结合完整对话理解您的问题并拆解查询任务。",
            )
            history_text = self._history_text(history)
            classification_started = time.perf_counter()
            child_tasks = await self.dependencies.classifier.classify(
                user_message.strip(),
                history_text,
            )
            classification_duration_ms = (time.perf_counter() - classification_started) * 1000

            route = self.dependencies.router.decide(child_tasks)
            yield StreamEvent(
                type=StreamEventType.ROUTE,
                content=f"问题已拆解为 {len(route.tasks)} 个子任务。",
                data={
                    "classification": {
                        "tasks": [task.model_dump(mode="json") for task in child_tasks],
                    },
                    "route": route.model_dump(mode="json"),
                    "duration_ms": classification_duration_ms,
                },
            )
            yield StreamEvent(
                type=StreamEventType.REASONING_SUMMARY,
                content=route.reason,
                data={
                    "categories": [task.category.value for task in route.tasks],
                    "requires_fresh_data": any(task.requires_fresh_data for task in route.tasks),
                },
            )

            if route.action == "clarify":
                answer = self.dependencies.router.clarification_message(child_tasks)
                async for event in self._answer_events(answer):
                    yield event
                return

            if route.action == "general_answer":
                yield StreamEvent(
                    type=StreamEventType.PROGRESS,
                    content="这是通用问题，不需要查询专业数据库，正在直接组织回答。",
                )
                answer = await self.dependencies.general_workflow.run(
                    user_message,
                    history_text,
                )
                async for event in self._answer_events(answer):
                    yield event
                return

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="正在按子任务执行对应工作流，并共享依赖任务的结论和证据。",
            )
            progress_queue = asyncio.Queue()
            progress_callback = progress_queue.put
            workflow_task = asyncio.create_task(
                self._execute_route_plan(
                    route,
                    user_message,
                    history_text,
                    progress_callback,
                )
            )
            try:
                while not workflow_task.done() or not progress_queue.empty():
                    try:
                        tool_event = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    yield StreamEvent(
                        type=StreamEventType.PROGRESS,
                        content=tool_event.summary,
                        data=tool_event.model_dump(mode="json"),
                    )
                result = await workflow_task
            finally:
                if not workflow_task.done():
                    workflow_task.cancel()
                    await asyncio.gather(workflow_task, return_exceptions=True)
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
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content=f"已核验 {len(result.evidence)} 条资料并完成引用检查，正在整理最终回答。",
            )
            async for event in self._answer_events(result.answer):
                yield event
        except AgentError as exc:
            logger.warning("agent run failed | code=%s", exc.code, exc_info=True)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=exc.user_message,
            )
        except Exception:
            logger.exception("unexpected agent failure")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="系统已记录错误编号。",
            )

    async def _answer_events(self, answer: str) -> AsyncIterator[StreamEvent]:
        size = self.dependencies.settings.stream_chunk_size
        for index in range(0, len(answer), size):
            yield StreamEvent(
                type=StreamEventType.ASSISTANT_DELTA,
                content=answer[index : index + size],
            )

    async def _execute_route_plan(
            self,
            route: RoutePlan,
            original_question: str,
            history: str,
            progress_callback,
    ) -> WorkflowResult:
        task_ids = {task.task_id for task in route.tasks}
        for task in route.tasks:
            if any(dependency not in task_ids for dependency in task.depends_on):
                raise ClassificationError

        pending = [task for task in route.tasks if task.category != Category.UNCLEAR]
        completed = set()
        evidence = []
        tool_events = []
        child_answers = []
        model_calls = 0
        while pending:
            ready = [task for task in pending if set(task.depends_on).issubset(completed)]
            if not ready:
                raise ClassificationError
            results = await asyncio.gather(
                *(
                    self._run_child_task(
                        task,
                        original_question,
                        history,
                        child_answers,
                        progress_callback,
                    )
                    for task in ready
                )
            )
            for task, result in zip(ready, results, strict=True):
                child_answers.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category.value,
                        "question": task.question,
                        "answer": result.answer,
                    }
                )
                evidence.extend(result.evidence)
                tool_events.extend(result.tool_events)
                model_calls += result.model_calls
                completed.add(task.task_id)
            ready_ids = {task.task_id for task in ready}
            pending = [task for task in pending if task.task_id not in ready_ids]

        synthesis = await self.dependencies.composite_workflow.run(
            original_question,
            history,
            child_answers,
            evidence,
        )
        tool_events.extend(synthesis.tool_events)
        return WorkflowResult(
            answer=synthesis.answer,
            evidence=synthesis.evidence,
            citations=synthesis.citations,
            tool_events=tool_events,
            model_calls=model_calls + synthesis.model_calls,
        )

    async def _run_child_task(
            self,
            task: ChildTask,
            original_question: str,
            history: str,
            child_answers: list[dict],
            progress_callback,
    ) -> WorkflowResult:
        task_question = self._task_question(task, child_answers)
        if task.category == Category.OTHER:
            answer = await self.dependencies.general_workflow.run(task_question, history)
            return WorkflowResult(answer=answer, model_calls=1)
        if task.category == Category.POLICY:
            return await self.dependencies.policy_workflow.run(
                original_question,
                task_question,
                history,
                progress_callback,
            )
        workflow = self.dependencies.data_workflows[task.category]
        return await workflow.run(
            task_question,
            history,
            [],
            task.requires_fresh_data,
            progress_callback,
        )

    @staticmethod
    def _task_question(task: ChildTask, child_answers: list[dict]) -> str:
        related = [item for item in child_answers if item["task_id"] in task.depends_on]
        if not related:
            return task.question
        context_lines = "\n".join(
            f"- {item['question']}：{item['answer']}"
            for item in related
        )
        return f"{task.question}\n\n已完成的相关子任务结论：\n{context_lines}"

    def _history_text(self, history_messages: list[Message]) -> str:
        messages = history_messages[-self.dependencies.settings.recent_history_messages:]
        return json.dumps(
            [
                {
                    "role": message.role,
                    "content": message.content[: self.dependencies.settings.context_message_chars],
                }
                for message in messages
            ],
            ensure_ascii=False,
        )

    def _quick_response(self, question: str) -> dict | None:
        normalized = re.sub(r"[\W_]+", "", question.strip().lower(), flags=re.UNICODE)
        if not normalized:
            return EMPTY_QUICK_RESPONSE
        return QUICK_RESPONSES.get(normalized)

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
            Category.UNCLEAR: "需要澄清",
        }[category]
