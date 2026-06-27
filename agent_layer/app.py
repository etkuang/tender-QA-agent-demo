# coding: utf-8

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from common.api_contracts.agent_api import Message
from common.logger import get_logger
from agent_layer.classification.chain import QuestionClassifier
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.config import Settings
from agent_layer.conversation.quick_responses import EMPTY_QUICK_RESPONSE, QUICK_RESPONSES
from agent_layer.errors import AgentError, DecompositionError
from agent_layer.schemas import (
    Category,
    ChildTask,
    ChildTaskOutcome,
    StreamEvent,
    StreamEventType,
    TaskStatus,
    ToolEvent,
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


class TaskGraph:
    def __init__(self, tasks: list[ChildTask]) -> None:
        task_ids = [task.task_id for task in tasks]
        task_map = {task.task_id: task for task in tasks}
        dependencies = {task.task_id: task.depends_on.copy() for task in tasks}

        if len(set(task_ids)) != len(task_ids):
            raise DecompositionError("question decomposer returned duplicate child task ids")

        if any(dependency not in task_ids for task in tasks for dependency in task.depends_on):
            raise DecompositionError("question decomposer returned invalid child task dependencies")

        self._task_map = task_map
        self._dependencies = dependencies
        self._frontier_task_ids = [
            task_id
            for task_id, task_dependencies in dependencies.items()
            if not task_dependencies
        ]

    @property
    def task_ids(self) -> list[str]:
        return list(self._task_map)

    @property
    def task_map(self) -> dict[str, ChildTask]:
        return self._task_map.copy()

    @property
    def dependencies(self) -> dict[str, list[str]]:
        return {
            task_id: task_dependencies.copy()
            for task_id, task_dependencies in self._dependencies.items()
        }

    @property
    def frontier_task_ids(self) -> list[str]:
        return self._frontier_task_ids.copy()


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
        user_message = user_message.strip()
        try:
            # ----- Step 1: quick response -----
            if user_message:
                quick_decision = await self.dependencies.quick_classifier.classify(user_message)
                quick = QUICK_RESPONSES.get(quick_decision.quick_response_type)
            else:
                quick = EMPTY_QUICK_RESPONSE

            if quick:
                yield StreamEvent(
                    type=StreamEventType.ROUTE,
                    content=quick["route_content"],
                )
                async for event in self._answer_events(quick["content"]):
                    yield event
                return

            # ----- Step 2: decompose user request -----
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="我正在参考最近对话，识别可以单独处理的子问题。",
            )

            history_text = self._history_text(history)
            child_tasks = await self.dependencies.classifier.classify(
                user_message,
                history_text,
            )

            # ----- Step 3: build task graph -----
            task_graph = TaskGraph(child_tasks)

            yield StreamEvent(
                type=StreamEventType.ROUTE,
                content=self._task_graph_content(task_graph),
            )

            yield StreamEvent(
                type=StreamEventType.REASONING_SUMMARY,
                content=self._execution_strategy_content(task_graph),
            )

            # ----- Step 4: execute child-task graph -----
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="我正在按依赖关系执行可处理的子任务。",
            )
            progress_queue = asyncio.Queue()
            progress_callback = progress_queue.put
            workflow_task = asyncio.create_task(
                self._execute_child_tasks(
                    task_graph,
                    user_message,
                    history_text,
                    progress_callback,
                )
            )
            try:
                # ----- Step 5: relay workflow progress -----
                while not workflow_task.done() or not progress_queue.empty():
                    try:
                        tool_event = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    yield StreamEvent(
                        type=StreamEventType.PROGRESS,
                        content=self._progress_content(tool_event),
                    )
                result = await workflow_task
            finally:
                if not workflow_task.done():
                    workflow_task.cancel()
                    await asyncio.gather(workflow_task, return_exceptions=True)
            for item in result.evidence:
                # ----- Step 6: stream sources and final answer -----
                yield StreamEvent(
                    type=StreamEventType.SOURCE,
                    content=self._source_content(item),
                )
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content=f"子任务处理完成，已整理 {len(result.evidence)} 条可引用资料，下面返回最终回答。",
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
                content="处理过程中出现未预期错误，请稍后重试。",
            )

    @staticmethod
    def _task_graph_content(task_graph: TaskGraph) -> str:
        category_labels = {
            Category.POLICY: "政策法规",
            Category.TENDER: "招标项目",
            Category.PUBLIC_OPINION: "舆情信息",
            Category.COMPANY: "企业信息",
            Category.PRICE: "价格信息",
            Category.PRODUCT: "商品信息",
            Category.OTHER: "通用问题",
            Category.UNCLEAR: "需要澄清",
        }
        task_ids = task_graph.task_ids
        task_map = task_graph.task_map
        dependencies_by_task = task_graph.dependencies
        task_descriptions = []
        for task_id in task_ids:
            task = task_map[task_id]
            dependencies = dependencies_by_task[task_id]
            dependency_text = ""
            if dependencies:
                dependency_text = f"，依赖 {', '.join(dependencies)}"
            freshness_text = ""
            if task.requires_fresh_data:
                freshness_text = "，需要较新或实时数据"
            task_descriptions.append(
                f"{task.task_id}：{task.question}（{category_labels[task.category]}{dependency_text}{freshness_text}）"
            )
        return f"已将问题拆成 {len(task_ids)} 个子任务，并建立依赖处理图：{'；'.join(task_descriptions)}。"

    @staticmethod
    def _execution_strategy_content(task_graph: TaskGraph) -> str:
        category_labels = {
            Category.POLICY: "政策法规",
            Category.TENDER: "招标项目",
            Category.PUBLIC_OPINION: "舆情信息",
            Category.COMPANY: "企业信息",
            Category.PRICE: "价格信息",
            Category.PRODUCT: "商品信息",
            Category.OTHER: "通用问题",
            Category.UNCLEAR: "需要澄清",
        }
        task_map = task_graph.task_map
        tasks = [task_map[task_id] for task_id in task_graph.task_ids]
        categories = "、".join(dict.fromkeys(category_labels[task.category] for task in tasks))
        freshness = "其中有子任务需要较新或实时数据。" if any(
            task.requires_fresh_data for task in tasks
        ) else "这些子任务未表达实时数据需求。"
        return (
            f"本轮包含{categories}类型的子任务，{freshness}"
            "我会先处理没有前置依赖的子任务；如果某个子任务无法解决，"
            "它的后续依赖任务会标记为无法继续，并在最终回答中说明。"
        )

    @staticmethod
    def _source_content(item) -> str:
        category_labels = {
            "policy": "政策法规",
            "tender": "招标项目",
            "public_opinion": "舆情信息",
            "company": "企业信息",
            "price": "价格信息",
            "product": "商品信息",
            "other": "通用问题",
            "unclear": "需要澄清",
        }
        source_type_labels = {
            "local_document": "本地资料库",
            "sql": "结构化数据库",
            "website": "外部网站",
        }
        domain = category_labels[item.domain.value]
        source_type = source_type_labels[item.source_type.value]
        legal_position = ""
        if item.law_name:
            legal_position = f"，对应《{item.law_name}》"
        if item.article_id:
            legal_position += f"第 {item.article_id} 条"
        published_at = ""
        if item.published_at:
            published_at = f"，发布日期：{item.published_at.date().isoformat()}"
        location = ""
        if item.url:
            location = f"，来源链接：{item.url}"
        elif item.document_id:
            location = f"，文档编号：{item.document_id}"
        return f"已将{domain}领域的{source_type}资料“{item.title}”{legal_position}{published_at}{location}作为本次回答依据。"

    @staticmethod
    def _progress_content(event: ToolEvent) -> str:
        details = event.details
        if event.stage == "policy_parse":
            if event.status == "started":
                return "正在提取政策检索条件。"
            if event.status == "completed":
                return "已提取政策检索条件。"
        if event.stage == "policy_retrieve":
            if event.status == "started":
                round_index = details.get("round")
                if round_index:
                    return f"正在进行第 {round_index} 轮政策资料检索。"
                return "正在检索政策资料。"
            if event.status == "completed":
                return f"本轮政策知识库检索返回 {details.get('evidence_count', 0)} 条候选资料。"
        if event.stage == "policy_assessment":
            if event.status == "started":
                return "正在判断当前政策资料是否足以回答。"
            if event.status == "completed":
                return "已完成政策资料充分性判断。"
        if event.stage == "policy_internet":
            if event.status == "skipped":
                return "未配置官方政策网页检索，本轮不查询外部网页。"
            if event.status == "failed":
                return "官方政策网页检索暂时不可用，本轮只使用已取得资料。"
            if event.status == "completed":
                return f"官方政策网页检索返回 {details.get('evidence_count', 0)} 条可用资料。"
        if event.stage == "policy_synthesis":
            if event.status == "started":
                return "正在根据已选政策资料生成政策子任务回答。"
            if event.status == "completed":
                return "已完成政策子任务回答。"
        if event.stage == "research_plan":
            if event.status == "started":
                return "正在为当前子任务生成数据查询计划。"
            if event.status == "completed":
                return f"已生成 {len(details.get('tasks', []))} 个数据查询步骤。"
        if event.stage == "sql":
            if event.status == "skipped":
                return "结构化数据库未配置，本次无法执行 SQL 查询。"
            if event.status == "started":
                return "正在执行只读结构化查询。"
            if event.status == "failed":
                return event.summary
            if event.status == "completed":
                return f"只读结构化查询完成，返回 {details.get('row_count', 0)} 行。"
        if event.stage == "website":
            if event.status == "skipped":
                return "未配置该领域的网站数据源，本次跳过网站检索。"
            if event.status == "started":
                return "正在检索已配置的网站数据源。"
            if event.status in {"completed", "failed"}:
                return (
                    f"网站检索返回 {details.get('evidence_count', 0)} 条可用资料，"
                    f"{details.get('failed_adapter_count', 0)} 个适配器失败。"
                )
        if event.stage == "domain_synthesis":
            if event.status == "started":
                return "已汇总可用数据，正在核对统计口径、缺失项和引用。"
            if event.status == "completed":
                return "已完成当前数据子任务回答。"
        return event.summary

    async def _answer_events(self, answer: str) -> AsyncIterator[StreamEvent]:
        size = self.dependencies.settings.stream_chunk_size
        for index in range(0, len(answer), size):
            yield StreamEvent(
                type=StreamEventType.ASSISTANT_DELTA,
                content=answer[index : index + size],
            )

    async def _execute_child_tasks(
            self,
            task_graph: TaskGraph,
            original_question: str,
            history: str,
            progress_callback,
    ) -> WorkflowResult:
        task_ids = task_graph.task_ids
        pending = task_graph.task_map
        dependencies_by_task = task_graph.dependencies
        outcomes = {}
        evidence = []
        tool_events = []
        model_calls = 0

        while pending:
            ready = []
            blocked_now = []
            for task in list(pending.values()):
                dependencies = dependencies_by_task[task.task_id]

                dependency_outcomes = [
                    outcomes[dependency]
                    for dependency in dependencies
                    if dependency in outcomes
                ]
                if len(dependency_outcomes) != len(dependencies):
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
            outcomes[task_id].model_dump(mode="json")
            for task_id in task_ids
            if task_id in outcomes
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
