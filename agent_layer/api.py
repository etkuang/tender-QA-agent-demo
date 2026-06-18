# coding: utf-8

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from common.api_contracts.agent_api import AgentStreamRequest, AgentTransportChunk
from agent_layer.bootstrap import build_application
from agent_layer.config import settings
from agent_layer.schemas import StreamEvent, StreamEventType
from common.logger import get_logger, reset_request_id, set_request_id

logger = get_logger("agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.application = await build_application()
    try:
        yield
    finally:
        await app.state.application.aclose()


app = FastAPI(lifespan=lifespan)

CATEGORY_LABELS = {
    "policy": "政策法规",
    "tender": "招标项目",
    "public_opinion": "舆情信息",
    "company": "企业信息",
    "price": "价格信息",
    "product": "商品信息",
    "other": "通用问题",
    "unclear": "需要澄清",
}

SOURCE_TYPE_LABELS = {
    "local_document": "本地资料库",
    "sql": "结构化数据库",
    "website": "外部网站",
}


def to_transport_chunks(event: StreamEvent) -> list[AgentTransportChunk]:
    size = settings.stream_chunk_size

    def build_chunks(chunk_type: str, content: str) -> list[AgentTransportChunk]:
        if not content:
            return [AgentTransportChunk(type=chunk_type, content=content)]
        return [
            AgentTransportChunk(
                type=chunk_type,
                content=content[index : index + size],
            )
            for index in range(0, len(content), size)
        ]

    if event.type == StreamEventType.ASSISTANT_DELTA:
        return build_chunks("assistant", event.content)
    if event.type == StreamEventType.ERROR:
        return build_chunks("error", event.content)
    if event.type == StreamEventType.ROUTE:
        return build_chunks("reasoning", _format_route(event))
    if event.type == StreamEventType.REASONING_SUMMARY:
        return build_chunks("reasoning", f"- 处理策略：{event.content}\n\n")
    if event.type == StreamEventType.PROGRESS:
        return build_chunks("reasoning", _format_progress(event))
    if event.type == StreamEventType.SOURCE:
        return build_chunks("reasoning", _format_source(event))
    return []


def _format_route(event: StreamEvent) -> str:
    classification = event.data.get("classification")
    if not classification:
        return f"- 问题判断：{event.content}\n\n"
    tasks = classification.get("tasks", [])
    if not tasks:
        return f"- 问题判断：{classification['reasoning']}\n\n"
    task_text = "；".join(
        f"{task['task_id']}“{task['question']}”归为{CATEGORY_LABELS[task['category']]}"
        for task in tasks
    )
    freshness = "其中包含需要较新数据的子问题" if classification["requires_fresh_data"] else "不要求实时数据"
    return f"- 问题拆解：{task_text}。{freshness}。总体依据是：{classification['reasoning']}\n\n"


def _format_progress(event: StreamEvent) -> str:
    details = event.data.get("details", {})
    if event.data.get("stage") == "policy_assessment" and details:
        conclusion = "现有资料足以支持回答" if details.get("sufficient") else "现有资料还不足，需要继续补充"
        reason = details.get("reason", "")
        return f"- 证据检查：{conclusion}。{reason}\n\n"
    if event.data.get("stage") == "research_plan" and details.get("tasks"):
        goals = "；".join(task["goal"] for task in details["tasks"])
        return f"- 查询计划：{event.content}准备依次处理：{goals}。\n\n"
    return f"- 处理进度：{event.content}\n\n"


def _format_source(event: StreamEvent) -> str:
    source_type = SOURCE_TYPE_LABELS[event.data["source_type"]]
    title = event.data["title"]
    law_name = event.data.get("law_name")
    article_id = event.data.get("article_id")
    legal_position = ""
    if law_name:
        legal_position = f"，对应《{law_name}》"
    if article_id:
        legal_position += f"第 {article_id} 条"
    return f"- 参考资料：已从{source_type}确认“{title}”{legal_position}。\n\n"


@app.post("/chat/stream")
async def stream_chat(req: AgentStreamRequest, request: Request):
    request_id = request.headers.get("X-Request-ID")
    if request_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Request-ID header")

    async def ndjson_stream():
        token = set_request_id(request_id)
        try:
            logger.info("agent.stream start | history_messages=%s", len(req.history_messages))
            async for event in request.app.state.application.stream(
                req.user_message,
                req.history_messages,
            ):
                if await request.is_disconnected():
                    logger.info("agent.stream cancelled by client")
                    break
                for chunk in to_transport_chunks(event):
                    payload = chunk.model_dump(mode="json", exclude_none=True)
                    yield f"{json.dumps(payload, ensure_ascii=False)}\n".encode("utf-8")
            logger.info("agent.stream finished")
        finally:
            reset_request_id(token)

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
