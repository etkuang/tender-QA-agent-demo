# coding: utf-8
# @Author: Wang Qingkang

import json
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent_layer.agent_state_schemas import AgentInputState, AgentOutputState, AgentState
from agent_layer.config import settings
from agent_layer.miscellaneous import EMPTY_QUICK_RESPONSE, QUICK_RESPONSES
from agent_layer.models import get_base_model
from agent_layer.prompt_templates import (
    QUESTION_DECOMPOSITION_PROMPT,
    QUICK_RESPONSE_PROMPT,
)
from agent_layer.structured_output_schemas import (
    ChildTask,
    ChildTaskList,
    QuickResponseDecision,
    QuickResponseType,
)


def prepare_request(state: AgentInputState) -> dict[str, str]:
    return {"user_message": state.user_message.strip()}


async def resolve_quick_response(
    state: AgentState,
    runtime: Runtime,
) -> Command[Literal["decompose_user_request", "__end__"]]:
    if not state.user_message:
        runtime.stream_writer(
            {
                "type": "reasoning",
                "content": "当前输入为空，我会提示您补充具体问题。",
            }
        )
        return Command(
            update={"answer": EMPTY_QUICK_RESPONSE},
            goto=END,
        )

    structured_model = get_base_model().with_structured_output(
        QuickResponseDecision,
        method="json_mode",
    )
    chain = QUICK_RESPONSE_PROMPT | structured_model
    decision = await chain.with_retry(
        stop_after_attempt=settings.structured_output_retries + 1,
    ).ainvoke(
        {
            "last_user_message": state.user_message,
        }
    )
    if decision.quick_response_type == QuickResponseType.NONE:
        return Command(goto="decompose_user_request")

    quick_response = QUICK_RESPONSES[decision.quick_response_type]
    runtime.stream_writer(
        {
            "type": "reasoning",
            "content": (
                "识别到用户意图仅为"
                f"{quick_response['intent_label']}"
                "，可直接快速回复。"
            ),
        }
    )
    return Command(
        update={"answer": quick_response["content"]},
        goto=END,
    )


async def decompose_user_request(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, list[ChildTask]]:
    runtime.stream_writer(
        {
            "type": "reasoning",
            "content": "我正在参考最近对话，识别可以单独处理的子问题。",
        }
    )
    history = json.dumps(
        [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in state.history_messages
        ],
        ensure_ascii=False,
    )
    structured_model = get_base_model().with_structured_output(
        ChildTaskList,
        method="json_mode",
    )
    chain = QUESTION_DECOMPOSITION_PROMPT | structured_model
    child_task_list = await chain.with_retry(
        stop_after_attempt=settings.structured_output_retries + 1,
    ).ainvoke(
        {
            "question": state.user_message,
            "history": history,
        }
    )
    return {"child_tasks": child_task_list.child_tasks}


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(
        state_schema=AgentState,
        input_schema=AgentInputState,
        output_schema=AgentOutputState,
    )
    builder.add_node(prepare_request)
    builder.add_node(resolve_quick_response)
    builder.add_node(decompose_user_request)
    builder.add_edge(START, "prepare_request")
    builder.add_edge("prepare_request", "resolve_quick_response")
    builder.add_edge("decompose_user_request", END)
    return builder.compile()