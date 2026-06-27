# coding: utf-8

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from common.api_contracts.agent_api import Message
from common.logger import get_logger
from agent_layer.classification.chain import QuestionClassifier
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.config import Settings
from agent_layer.conversation.quick_responses import EMPTY_QUICK_RESPONSE, QUICK_RESPONSES
from agent_layer.errors import AgentError, ClassificationError
from agent_layer.schemas import (
    Category,
    ChildTask,
    ChildTaskOutcome,
    StreamEvent,
    StreamEventType,
    TaskStatus,
    WorkflowResult,
)
from agent_layer.workflows.data_domain import DataDomainWorkflow
from agent_layer.workflows.general import CompositeAnswerWorkflow, GeneralWorkflow
from agent_layer.workflows.policy_graph import PolicyGraphWorkflow

logger = get_logger("agent.app")


@dataclass(frozen=True)
class ApplicationDependencies:
    settings: Settings
    quick_classifier: QuickResponseClassifier
    classifier: QuestionClassifier
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
            stripped_message = user_message.strip()
            if stripped_message:
                quick_decision = await self.dependencies.quick_classifier.classify(stripped_message)
                quick = QUICK_RESPONSES.get(quick_decision.quick_response_type)
            else:
                quick = EMPTY_QUICK_RESPONSE

            if quick:
                yield StreamEvent(
                    type=StreamEventType.ROUTE,
                    content="已识别为快捷回复消息。",
                    data={"route": quick["kind"]},
                )
                async for event in self._answer_events(quick["content"]):
                    yield event
                return

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="正在结合最近对话理解您的问题并拆解查询任务。",
            )
            history_text = self._history_text(history)
            classification_started = time.perf_counter()
            child_tasks = await self.dependencies.classifier.classify(
                stripped_message,
                history_text,
            )
            classification_duration_ms = (time.perf_counter() - classification_started) * 1000

            yield StreamEvent(
                type=StreamEventType.ROUTE,
                content=f"问题已拆解为 {len(child_tasks)} 个子任务。",
                data={
                    "classification": {
                        "tasks": [task.model_dump(mode="json") for task in child_tasks],
                    },
                    "duration_ms": classification_duration_ms,
                },
            )
            yield StreamEvent(
                type=StreamEventType.REASONING_SUMMARY,
                content="问题已拆成子任务；我会按依赖关系执行可处理任务，并说明无法解决或被依赖阻塞的任务。",
                data={
                    "categories": [task.category.value for task in child_tasks],
                    "requires_fresh_data": any(task.requires_fresh_data for task in child_tasks),
                },
            )

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="正在按子任务依赖关系执行可处理任务。",
            )
            progress_queue = asyncio.Queue()
            progress_callback = progress_queue.put
            workflow_task = asyncio.create_task(
                self._execute_child_tasks(
                    child_tasks,
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

    async def _execute_child_tasks(
            self,
            tasks: list[ChildTask],
            original_question: str,
            history: str,
            progress_callback,
    ) -> WorkflowResult:
        pending = {task.task_id: task for task in tasks}
        outcomes = {}
        evidence = []
        tool_events = []
        model_calls = 0

        while pending:
            ready = []
            blocked_now = []
            for task in list(pending.values()):
                missing_dependencies = [
                    dependency
                    for dependency in task.depends_on
                    if dependency not in pending and dependency not in outcomes
                ]
                if missing_dependencies:
                    blocked_now.append(
                        (
                            task,
                            f"依赖的子任务不存在：{', '.join(missing_dependencies)}。",
                        )
                    )
                    continue

                dependency_outcomes = [
                    outcomes[dependency]
                    for dependency in task.depends_on
                    if dependency in outcomes
                ]
                if len(dependency_outcomes) != len(task.depends_on):
                    continue

                unsolved_dependencies = [
                    outcome
                    for outcome in dependency_outcomes
                    if outcome.status != TaskStatus.SOLVED
                ]
                if unsolved_dependencies:
                    dependency_ids = ", ".join(outcome.task_id for outcome in unsolved_dependencies)
                    blocked_now.append((task, f"依赖的子任务未解决：{dependency_ids}。"))
                    continue

                ready.append(task)

            for task, reason in blocked_now:
                outcomes[task.task_id] = self._blocked_outcome(task, reason)
                pending.pop(task.task_id)

            if not ready:
                if not blocked_now:
                    for task in list(pending.values()):
                        outcomes[task.task_id] = self._blocked_outcome(
                            task,
                            "子任务依赖关系存在循环，无法确定执行顺序。",
                        )
                        pending.pop(task.task_id)
                continue

            results = await asyncio.gather(
                *(
                    self._run_child_task(
                        task,
                        original_question,
                        history,
                        list(outcomes.values()),
                        progress_callback,
                    )
                    for task in ready
                )
            )
            for task, result in zip(ready, results, strict=True):
                outcome = self._outcome_from_result(task, result)
                outcomes[task.task_id] = outcome
                evidence.extend(result.evidence)
                tool_events.extend(result.tool_events)
                model_calls += result.model_calls
                pending.pop(task.task_id)

        child_results = [
            outcomes[task.task_id].model_dump(mode="json")
            for task in tasks
            if task.task_id in outcomes
        ]
        synthesis = await self.dependencies.composite_workflow.run(
            original_question,
            history,
            child_results,
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
            child_results: list[ChildTaskOutcome],
            progress_callback,
    ) -> WorkflowResult:
        if task.category == Category.UNCLEAR:
            reason = task.clarification_question
            return WorkflowResult(
                answer=reason,
                status=TaskStatus.UNSOLVED,
                unresolved_reason=reason,
            )

        task_question = self._task_question(task, child_results)
        try:
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
        except AgentError as exc:
            return WorkflowResult(
                answer=exc.user_message,
                status=TaskStatus.UNSOLVED,
                unresolved_reason=exc.user_message,
            )

    @staticmethod
    def _outcome_from_result(task: ChildTask, result: WorkflowResult) -> ChildTaskOutcome:
        reason = result.unresolved_reason
        return ChildTaskOutcome(
            task_id=task.task_id,
            category=task.category,
            question=task.question,
            depends_on=task.depends_on,
            status=result.status,
            answer=result.answer,
            reason=reason,
            evidence=result.evidence,
            citations=result.citations,
            tool_events=result.tool_events,
            model_calls=result.model_calls,
        )

    @staticmethod
    def _blocked_outcome(task: ChildTask, reason: str) -> ChildTaskOutcome:
        return ChildTaskOutcome(
            task_id=task.task_id,
            category=task.category,
            question=task.question,
            depends_on=task.depends_on,
            status=TaskStatus.BLOCKED,
            answer=reason,
            reason=reason,
        )

    @staticmethod
    def _task_question(task: ChildTask, child_results: list[ChildTaskOutcome]) -> str:
        related = [
            item
            for item in child_results
            if item.task_id in task.depends_on and item.status == TaskStatus.SOLVED
        ]
        if not related:
            return task.question
        context_lines = "\n".join(
            f"- {item.question}：{item.answer}"
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
