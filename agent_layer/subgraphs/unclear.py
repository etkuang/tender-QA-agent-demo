# coding: utf-8
# @Author: Wang Qingkang

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import Field

from agent_layer.child_task_state_schemas import (
    ChildTaskInputState,
    ChildTaskOutput,
    ChildTaskOutputState,
)
from agent_layer.structured_output_schemas import AnswerStatus


class UnclearInputState(ChildTaskInputState):
    clarification_question: str = Field(min_length=1)


class UnclearState(UnclearInputState):
    child_task_output: ChildTaskOutput | None = None


def resolve_clarification(
    state: UnclearInputState,
    runtime: Runtime,
) -> dict[str, ChildTaskOutput]:
    runtime.stream_writer(
        {
            "type": "reasoning",
            "content": "当前问题的信息还不足，我会请求补充必要条件。",
        }
    )
    return {
        "child_task_output": ChildTaskOutput(
            task_id=state.task_id,
            status=AnswerStatus.NEEDS_CLARIFICATION,
            answer=state.clarification_question,
            evidence_ids=[],
        )
    }


def build_unclear_subgraph() -> CompiledStateGraph:
    builder = StateGraph(
        state_schema=UnclearState,
        input_schema=UnclearInputState,
        output_schema=ChildTaskOutputState,
    )
    builder.add_node(resolve_clarification)
    builder.add_edge(START, "resolve_clarification")
    builder.add_edge("resolve_clarification", END)
    return builder.compile(
        checkpointer=None,
        name="unclear",
    )