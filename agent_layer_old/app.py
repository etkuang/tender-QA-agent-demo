# coding: utf-8

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from common.api_contracts.agent_api import Message
from common.logger import get_logger, get_request_id
from agent_layer_old.question_decomposition.chain import QuestionDecomposer
from agent_layer_old.checkpoint import LangGraphCheckpointRuntime
from agent_layer_old.conversation.quick_classifier import QuickResponseClassifier
from agent_layer_old.config import Settings
from agent_layer_old.conversation.quick_responses import EMPTY_QUICK_RESPONSE, QUICK_RESPONSES
from agent_layer_old.errors import AgentError, DecompositionError
from agent_layer_old.schemas import (
    Category,
    ChildTask,
    ChildTaskOutcome,
    DependencyOutcome,
    StreamEvent,
    StreamEventType,
    ToolEvent,
    WorkflowResult,
)
from agent_layer_old.workflows.child_task import GeneralChildTaskWorkflow
from agent_layer_old.workflows.general import CompositeAnswerWorkflow
from agent_layer_old.workflows.policy_graph import PolicyWorkflow

logger = get_logger("agent.app")


@dataclass(frozen=True)
class ApplicationDependencies:
    settings: Settings
    quick_classifier: QuickResponseClassifier
    decomposer: QuestionDecomposer
    child_task_workflows: dict[Category, PolicyWorkflow | GeneralChildTaskWorkflow]
    composite_workflow: CompositeAnswerWorkflow
    checkpoint_runtime: LangGraphCheckpointRuntime


class TaskGraph:
    def __init__(self, tasks: list[ChildTask]) -> None:
        task_ids = [task.task_id for task in tasks]

        if len(set(task_ids)) != len(task_ids):
            raise DecompositionError("question decomposer returned duplicate child task ids")

        task_map = {task.task_id: task for task in tasks}
        dependencies = {task.task_id: set(task.depends_on) for task in tasks}

        if any(dependency not in task_map for task in tasks for dependency in task.depends_on):
            raise DecompositionError("question decomposer returned invalid child task dependencies")

        dependents, dependency_counts, frontier_task_ids = {task_id: [] for task_id in task_ids}, {}, []
        for task_id, task_dependencies in dependencies.items():
            if task_dependencies:
                dependency_counts[task_id] = len(task_dependencies)
            else:
                frontier_task_ids.append(task_id)
            for dependency in task_dependencies:
                dependents[dependency].append(task_id)

        ready = frontier_task_ids.copy()
        visited_count = 0

        while ready:
            task_id = ready.pop()
            visited_count += 1
            for dependent_id in dependents[task_id]:
                dependency_counts[dependent_id] -= 1
                if dependency_counts[dependent_id] == 0:
                    ready.append(dependent_id)

        if visited_count != len(task_ids):
            raise DecompositionError("question decomposer returned cyclic child task dependencies")

        self._task_map = task_map
        self._dependencies = dependencies
        self._dependents = dependents
        self._frontier_task_ids = frontier_task_ids

    @property
    def task_ids(self) -> list[str]:
        return list(self._task_map)

    @property
    def task_map(self) -> dict[str, ChildTask]:
        return self._task_map.copy()

    @property
    def dependencies(self) -> dict[str, set[str]]:
        return {
            task_id: task_dependencies.copy()
            for task_id, task_dependencies in self._dependencies.items()
        }

    @property
    def dependents(self) -> dict[str, list[str]]:
        return {
            task_id: dependent_ids.copy()
            for task_id, dependent_ids in self._dependents.items()
        }

    @property
    def frontier_task_ids(self) -> list[str]:
        return self._frontier_task_ids.copy()


class StreamEventFormatter:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def task_graph_content(task_graph: TaskGraph) -> str:
        category_labels = {
            Category.POLICY: "政策法规",
            Category.TENDER: "招标项目",
            Category.PUBLIC_OPINION: "舆情信息",
            Category.COMPANY: "企业信息",
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
                dependency_text = f"，依赖 {', '.join(sorted(dependencies))}"
            freshness_text = ""
            if task.requires_fresh_data:
                freshness_text = "，需要较新或实时数据"
            task_descriptions.append(
                f"{task.task_id}：{task.question}（{category_labels[task.category]}{dependency_text}{freshness_text}）"
            )
        return f"已将问题拆成 {len(task_ids)} 个子任务，并建立依赖处理图：{'；'.join(task_descriptions)}。"

    @staticmethod
    def execution_strategy_content(task_graph: TaskGraph) -> str:
        category_labels = {
            Category.POLICY: "政策法规",
            Category.TENDER: "招标项目",
            Category.PUBLIC_OPINION: "舆情信息",
            Category.COMPANY: "企业信息",
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
    def source_content(item) -> str:
        category_labels = {
            "policy": "政策法规",
            "tender": "招标项目",
            "public_opinion": "舆情信息",
            "company": "企业信息",
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
    def progress_content(event: ToolEvent) -> str:
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
        if event.stage == "general_internet":
            if event.status == "started":
                return "正在查询通用互联网资料。"
            if event.status == "skipped":
                return "通用互联网检索未配置，本次无法查询实时通用信息。"
            if event.status == "failed":
                return "通用互联网检索暂时不可用。"
            if event.status == "completed":
                return f"通用互联网检索返回 {details.get('evidence_count', 0)} 条可用资料。"
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

    async def answer_events(self, answer: str) -> AsyncIterator[StreamEvent]:
        size = self.settings.stream_chunk_size
        for index in range(0, len(answer), size):
            yield StreamEvent(
                type=StreamEventType.ASSISTANT_DELTA,
                content=answer[index : index + size],
            )


class ChildWorkflowRunner:
    def __init__(self, dependencies: ApplicationDependencies):
        self.dependencies = dependencies

    async def run(
        self,
        task: ChildTask,
        history: str,
        dependency_outcomes: list[DependencyOutcome],
        enqueue_progress_event: Callable[[ToolEvent], Awaitable[None]],
    ) -> ChildTaskOutcome:
        if task.category == Category.UNCLEAR:
            reason = task.clarification_question
            return ChildTaskOutcome(
                task_id=task.task_id,
                status="unsolved",
                answer=None,
                unresolved_reason=reason,
            )

        run_id = f"{get_request_id()}:{task.task_id}"

        async def enqueue_task_progress(event: ToolEvent) -> None:
            event.task_id = task.task_id
            await enqueue_progress_event(event)

        try:
            workflow = self.dependencies.child_task_workflows[task.category]
            result = await workflow.run(
                run_id,
                task.question,
                history,
                dependency_outcomes,
                task.requires_fresh_data,
                enqueue_task_progress,
            )
        except AgentError as exc:
            return ChildTaskOutcome(
                task_id=task.task_id,
                status="unsolved",
                answer=None,
                unresolved_reason=exc.user_message,
            )

        return ChildTaskOutcome(
            task_id=task.task_id,
            status=result.status,
            answer=result.answer if result.status == "solved" else None,
            unresolved_reason=result.unresolved_reason,
            evidence=result.evidence,
            citations=result.citations,
            tool_events=result.tool_events,
            model_calls=result.model_calls,
        )


class ChildTaskExecutor:
    def __init__(
        self,
        child_workflow_runner: ChildWorkflowRunner,
        composite_workflow: CompositeAnswerWorkflow,
    ):
        self.child_workflow_runner = child_workflow_runner
        self.composite_workflow = composite_workflow

    async def execute(
        self,
        task_graph: TaskGraph,
        original_question: str,
        history: str,
        enqueue_progress_event: Callable[[ToolEvent], Awaitable[None]],
    ) -> WorkflowResult:
        task_ids = task_graph.task_ids
        task_map = task_graph.task_map
        dependencies_by_task = task_graph.dependencies
        dependents_by_task = task_graph.dependents
        outcomes = {}
        evidence = []
        tool_events = []
        model_calls = 0
        pending = set(task_ids)
        running_executions = {}

        def start_child_task(child_task: ChildTask) -> None:
            dependency_outcomes = [
                DependencyOutcome(
                    task_id=dependency_id,
                    question=task_map[dependency_id].question,
                    answer=outcomes[dependency_id].answer,
                )
                for dependency_id in child_task.depends_on
            ]
            execution_task = asyncio.create_task(
                self.child_workflow_runner.run(
                    child_task,
                    history,
                    dependency_outcomes,
                    enqueue_progress_event,
                )
            )
            running_executions[execution_task] = child_task
            pending.remove(child_task.task_id)

        def block_child_task(child_task: ChildTask, reason: str) -> None:
            outcomes[child_task.task_id] = ChildTaskOutcome(
                task_id=child_task.task_id,
                status="blocked",
                answer=None,
                unresolved_reason=reason,
            )
            pending.remove(child_task.task_id)
            schedule_dependents(child_task.task_id)

        def schedule_dependents(task_id: str) -> None:
            outcome = outcomes[task_id]
            for dependent_id in dependents_by_task[task_id]:
                if dependent_id not in pending:
                    continue
                dependent_child_task = task_map[dependent_id]
                if outcome.status != "solved":
                    reason = f"依赖的子任务未解决：{task_id}。"
                    block_child_task(dependent_child_task, reason)
                    continue
                dependencies_by_task[dependent_id].remove(task_id)
                if dependencies_by_task[dependent_id]:
                    continue
                start_child_task(dependent_child_task)

        for task_id in task_graph.frontier_task_ids:
            start_child_task(task_map[task_id])

        while running_executions:
            finished, _ = await asyncio.wait(
                running_executions,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for execution_task in finished:
                child_task = running_executions.pop(execution_task)
                outcome = await execution_task
                outcomes[child_task.task_id] = outcome
                evidence.extend(outcome.evidence)
                tool_events.extend(outcome.tool_events)
                model_calls += outcome.model_calls
                schedule_dependents(child_task.task_id)

        child_results = []
        for task_id in task_ids:
            child_task = task_map[task_id]
            result = outcomes[task_id].model_dump(mode="json")
            result["category"] = child_task.category.value
            result["question"] = child_task.question
            result["depends_on"] = child_task.depends_on
            child_results.append(result)

        synthesis = await self.composite_workflow.run(
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


class TenderQAApplication:
    def __init__(self, dependencies: ApplicationDependencies):
        self.dependencies = dependencies
        self.formatter = StreamEventFormatter(dependencies.settings)
        self.child_task_executor = ChildTaskExecutor(
            ChildWorkflowRunner(dependencies),
            dependencies.composite_workflow,
        )

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
                async for event in self.formatter.answer_events(quick["content"]):
                    yield event
                return

            # ----- Step 2: decompose user request -----
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="我正在参考最近对话，识别可以单独处理的子问题。",
            )

            history_text = self._history_text(history)
            child_tasks = await self.dependencies.decomposer.decompose(
                user_message,
                history_text,
            )

            # ----- Step 3: build task graph -----
            task_graph = TaskGraph(child_tasks)

            yield StreamEvent(
                type=StreamEventType.ROUTE,
                content=self.formatter.task_graph_content(task_graph),
            )

            yield StreamEvent(
                type=StreamEventType.REASONING_SUMMARY,
                content=self.formatter.execution_strategy_content(task_graph),
            )

            # ----- Step 4: execute child-task graph -----
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content="我正在按依赖关系执行可处理的子任务。",
            )
            progress_queue = asyncio.Queue()
            workflow_task = asyncio.create_task(
                self.child_task_executor.execute(
                    task_graph,
                    user_message,
                    history_text,
                    progress_queue.put,
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
                        content=f"{tool_event.task_id}：{self.formatter.progress_content(tool_event)}",
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
                    content=self.formatter.source_content(item),
                )
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                content=f"子任务处理完成，已整理 {len(result.evidence)} 条可引用资料，下面返回最终回答。",
            )
            async for event in self.formatter.answer_events(result.answer):
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
