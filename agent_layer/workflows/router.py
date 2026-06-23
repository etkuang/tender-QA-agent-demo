# coding: utf-8


from agent_layer.schemas import Category, ChildTask, RoutePlan


class WorkflowRouter:
    @staticmethod
    def decide(tasks: list[ChildTask]) -> RoutePlan:
        if any(task.clarification_question is not None for task in tasks):
            return RoutePlan(
                action="clarify",
                tasks=tasks,
                reason="问题中仍有业务对象或查询意图不明确的部分，需要先澄清。",
            )

        if all(task.category == Category.OTHER for task in tasks):
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
            if task.clarification_question is not None
        ]
        return "；".join(questions)
