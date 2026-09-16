# coding: utf-8
# @Author: Wang Qingkang


from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import Field

from agent_layer.child_task_state_schemas import (
    ChildTaskInputState,
    ChildTaskOutput,
    ChildTaskOutputState,
)
from agent_layer.evidence_schemas import WebsiteEvidence
from agent_layer.integrations.internet_search import (
    INTERNET_SEARCH_TOOL_NAME,
    INTERNET_SEARCH_TOOL_SCHEMA,
    InternetSearchRequest,
    InternetSearchResult,
    InternetSearchStatus,
)
from agent_layer.prompt_templates import (
    OTHER_ANSWER_PROMPT,
    OTHER_RESEARCH_PROMPT,
)
from agent_layer.structured_output_schemas import (
    AnswerDraft,
    AnswerStatus,
)
from agent_layer.subgraphs.common import (
    ChildTaskRuntimeContext,
    format_evidence_markdown,
    format_history_markdown,
    format_prerequisite_answers_markdown,
    freshness_requirement,
    merge_evidence,
)

ResearchRoute = Literal["invoke_internet_search", "freeze_evidence"]
SearchContinuation = Literal["call_research_model", "freeze_evidence"]


class OtherState(ChildTaskInputState):
    pool_evidence: list[WebsiteEvidence] = Field(default_factory=list)
    prerequisite_evidence: list[WebsiteEvidence] = Field(default_factory=list)
    internet_evidence: list[WebsiteEvidence] = Field(default_factory=list)
    pending_search_result: InternetSearchResult | None = None
    research_messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list
    )
    search_rounds: int = 0
    search_round_limit: int = 0
    answer_evidence: list[WebsiteEvidence] = Field(default_factory=list)
    evidence_markdown: str = "无。"
    answer_draft: AnswerDraft | None = None
    child_task_output: ChildTaskOutput | None = None


def validate_research_response(message: BaseMessage) -> AIMessage:
    if not isinstance(message, AIMessage):
        raise ValueError("The research model must return an AIMessage")
    if message.invalid_tool_calls:
        raise ValueError("The research model returned invalid tool calls")
    if len(message.tool_calls) > 1:
        raise ValueError(
            "The research model may call Internet search once per round"
        )

    normalized_tool_calls = []
    for tool_call in message.tool_calls:
        if tool_call["name"] != INTERNET_SEARCH_TOOL_NAME:
            raise ValueError("The research model called an unsupported tool")
        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            raise ValueError("Internet search tool calls require an id")
        request = InternetSearchRequest.model_validate(tool_call["args"])
        query = " ".join(request.query.strip().split())
        if not query:
            raise ValueError("Internet search tool calls require a query")
        normalized_request = request.model_copy(
            update={"query": query}
        )
        normalized_tool_calls.append(
            {
                "name": INTERNET_SEARCH_TOOL_NAME,
                "args": normalized_request.model_dump(),
                "id": tool_call_id,
                "type": "tool_call",
            }
        )
    return message.model_copy(
        update={
            "tool_calls": normalized_tool_calls,
            "invalid_tool_calls": [],
        }
    )


def validate_answer_draft(
    draft: AnswerDraft,
    evidence: list[WebsiteEvidence],
    requires_fresh_data: bool,
) -> AnswerDraft:
    if draft.status == AnswerStatus.NEEDS_CLARIFICATION:
        raise ValueError(
            "The other answer model cannot request clarification"
        )
    visible_evidence_ids = {
        item.evidence_id
        for item in evidence
    }
    if not set(draft.evidence_ids).issubset(visible_evidence_ids):
        raise ValueError(
            "The answer cited evidence that was not visible in its prompt"
        )
    if (
        requires_fresh_data
        and draft.status == AnswerStatus.ANSWERED
        and not draft.evidence_ids
    ):
        raise ValueError(
            "Fresh-information answers must cite visible evidence"
        )
    return draft


async def retrieve_pool_evidence(
    state: ChildTaskInputState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, object]:
    runtime.stream_writer(
        {
            "type": "reasoning",
            "content": "我先从证据库检索与当前问题相关的资料。",
        }
    )
    settings = runtime.context.settings
    prerequisite_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for answer in state.prerequisite_answers
            for evidence_id in answer.evidence_ids
        )
    )
    pool_evidence = await runtime.context.evidence_pool.get_evidences(
        state.session_id,
        state.question,
        settings.evidence_pool_search_limit,
    )
    prerequisite_evidence = []
    for evidence_id in prerequisite_evidence_ids:
        evidence = await runtime.context.evidence_pool.get_evidence(
            state.session_id,
            evidence_id,
        )
        if evidence is not None:
            prerequisite_evidence.append(evidence)
    return {
        "pool_evidence": pool_evidence,
        "prerequisite_evidence": prerequisite_evidence,
        "search_round_limit": settings.internet_search_max_rounds,
    }


async def call_research_model(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, list[AIMessage]]:
    settings = runtime.context.settings
    _, evidence_markdown = format_evidence_markdown(
        merge_evidence(
            state.prerequisite_evidence,
            state.pool_evidence,
            state.internet_evidence,
        ),
        settings,
    )
    prompt_value = await OTHER_RESEARCH_PROMPT.ainvoke(
        {
            "question": state.question,
            "freshness_requirement": freshness_requirement(
                state.requires_fresh_data
            ),
            "history": format_history_markdown(
                state.history_messages,
            ),
            "prerequisite_answers": (
                format_prerequisite_answers_markdown(
                    state.prerequisite_answers,
                    settings,
                )
            ),
            "evidence": evidence_markdown,
        }
    )
    bound_model = runtime.context.research_model.bind_tools(
        [INTERNET_SEARCH_TOOL_SCHEMA],
        tool_choice="auto",
    )
    chain = bound_model | RunnableLambda(validate_research_response)
    message = await chain.with_retry(
        stop_after_attempt=settings.structured_output_retries + 1,
    ).ainvoke(
        [
            *prompt_value.to_messages(),
            *state.research_messages,
        ]
    )
    return {"research_messages": [message]}


def route_research(state: OtherState) -> ResearchRoute:
    latest_message = state.research_messages[-1]
    if (
        latest_message.tool_calls
        and state.search_rounds < state.search_round_limit
    ):
        return "invoke_internet_search"
    return "freeze_evidence"


async def invoke_internet_search(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, InternetSearchResult]:
    tool_call = state.research_messages[-1].tool_calls[0]
    request = InternetSearchRequest.model_validate(tool_call["args"])
    result = await runtime.context.internet_search_gateway.search(
        request.query,
        runtime.context.settings.internet_search_result_limit,
    )
    return {"pending_search_result": result}


def route_after_search(state: OtherState) -> SearchContinuation:
    if state.search_rounds < state.search_round_limit:
        return "call_research_model"
    return "freeze_evidence"


def freeze_evidence(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, object]:
    answer_evidence, evidence_markdown = format_evidence_markdown(
        merge_evidence(
            state.prerequisite_evidence,
            state.pool_evidence,
            state.internet_evidence,
        ),
        runtime.context.settings,
    )
    return {
        "answer_evidence": answer_evidence,
        "evidence_markdown": evidence_markdown,
    }


async def save_new_evidence(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, object]:
    result = state.pending_search_result
    saved_evidence = []
    if result.status == InternetSearchStatus.COMPLETED:
        if result.evidence:
            runtime.stream_writer(
                {
                    "type": "reasoning",
                    "content": "我正在把新取得的资料保存到证据库。",
                }
            )
        for draft in result.evidence:
            evidence = await runtime.context.evidence_pool.add_evidence(
                state.session_id,
                draft,
            )
            saved_evidence.append(evidence)
        evidence_ids = "、".join(
            evidence.evidence_id
            for evidence in saved_evidence
        )
        if evidence_ids:
            content = f"检索完成，取得证据：{evidence_ids}。"
        else:
            content = "检索完成，未找到相关公开资料。"
    else:
        content = f"检索失败：{result.reason}"

    tool_call = state.research_messages[-1].tool_calls[0]
    return {
        "internet_evidence": merge_evidence(
            state.internet_evidence,
            saved_evidence,
        ),
        "research_messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=INTERNET_SEARCH_TOOL_NAME,
            )
        ],
        "pending_search_result": None,
        "search_rounds": state.search_rounds + 1,
    }


async def generate_structured_answer(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, AnswerDraft]:
    if state.requires_fresh_data and not state.answer_evidence:
        return {
            "answer_draft": AnswerDraft(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer="当前问题需要最新信息，但没有可用于可靠回答的证据。",
                evidence_ids=[],
            )
        }

    runtime.stream_writer(
        {
            "type": "reasoning",
            "content": "我正在根据可用资料组织完整回答。",
        }
    )
    settings = runtime.context.settings
    structured_model = runtime.context.answer_model.with_structured_output(
        AnswerDraft,
        method="json_mode",
    )
    chain = (
        OTHER_ANSWER_PROMPT
        | structured_model
        | RunnableLambda(
            lambda draft: validate_answer_draft(
                draft,
                state.answer_evidence,
                state.requires_fresh_data,
            )
        )
    )
    answer_draft = await chain.with_retry(
        stop_after_attempt=settings.structured_output_retries + 1,
    ).ainvoke(
        {
            "question": state.question,
            "freshness_requirement": freshness_requirement(
                state.requires_fresh_data
            ),
            "history": format_history_markdown(
                state.history_messages,
            ),
            "prerequisite_answers": (
                format_prerequisite_answers_markdown(
                    state.prerequisite_answers,
                    settings,
                )
            ),
            "evidence": state.evidence_markdown,
        }
    )
    return {"answer_draft": answer_draft}


async def assemble_output(
    state: OtherState,
    runtime: Runtime[ChildTaskRuntimeContext],
) -> dict[str, ChildTaskOutput]:
    for evidence_id in state.answer_draft.evidence_ids:
        evidence = await runtime.context.evidence_pool.get_evidence(
            state.session_id,
            evidence_id,
        )
        if evidence is None:
            raise ValueError(
                "The answer cited evidence unavailable in this session"
            )
    return {
        "child_task_output": ChildTaskOutput(
            task_id=state.task_id,
            status=state.answer_draft.status,
            answer=state.answer_draft.answer,
            evidence_ids=state.answer_draft.evidence_ids,
        )
    }


def build_other_subgraph() -> CompiledStateGraph:
    builder = StateGraph(
        state_schema=OtherState,
        context_schema=ChildTaskRuntimeContext,
        input_schema=ChildTaskInputState,
        output_schema=ChildTaskOutputState,
    )
    builder.add_node(retrieve_pool_evidence)
    builder.add_node(call_research_model)
    builder.add_node(invoke_internet_search)
    builder.add_node(freeze_evidence)
    builder.add_node(save_new_evidence)
    builder.add_node(generate_structured_answer)
    builder.add_node(assemble_output)
    builder.add_edge(START, "retrieve_pool_evidence")
    builder.add_edge("retrieve_pool_evidence", "call_research_model")
    builder.add_conditional_edges(
        "call_research_model",
        route_research,
    )
    builder.add_edge("invoke_internet_search", "save_new_evidence")
    builder.add_conditional_edges(
        "save_new_evidence",
        route_after_search,
    )
    builder.add_edge("freeze_evidence", "generate_structured_answer")
    builder.add_edge("generate_structured_answer", "assemble_output")
    builder.add_edge("assemble_output", END)
    return builder.compile(
        checkpointer=None,
        name="other",
    )