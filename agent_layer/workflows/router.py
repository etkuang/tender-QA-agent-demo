# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category, ChildTask, RoutePlan


class WorkflowRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(self, tasks: list[ChildTask]) -> RoutePlan:
        if not tasks:
            return RoutePlan(
                action="clarify",
                reason="我还不能把当前问题拆成可执行的查询任务，需要先确认您的查询目标。",
            )

        if any(task.category == Category.UNCLEAR for task in tasks):
            return RoutePlan(
                action="clarify",
                tasks=tasks,
                reason="问题中仍有业务对象或查询意图不明确的部分，需要先澄清。",
            )

        executable_tasks = [task for task in tasks if task.category != Category.OTHER]
        if not executable_tasks:
            return RoutePlan(
                action="general_answer",
                tasks=tasks,
                reason="问题意图明确，但不需要调用招投标专业数据源，可以直接通用回答。",
            )

        return RoutePlan(
            action="execute",
            tasks=tasks,
            reason="问题已拆成明确的子任务，我会按任务类别调用对应工作流，并在最终回答中合并结论和证据。",
        )

    @staticmethod
    def clarification_message(tasks: list[ChildTask]) -> str:
        questions = [
            task.clarification_question
            for task in tasks
            if task.category == Category.UNCLEAR and task.clarification_question
        ]
        if questions:
            return "；".join(questions[:3])
        return "请补充您希望查询的业务对象、时间范围或最关心的结论。"
