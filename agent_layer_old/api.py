# coding: utf-8

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from common.api_contracts.agent_api import AgentStreamRequest, AgentTransportChunk
from agent_layer_old.bootstrap import build_application
from agent_layer_old.config import settings
from agent_layer_old.schemas import StreamEvent, StreamEventType
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
    if event.type in {
        StreamEventType.ROUTE,
        StreamEventType.REASONING_SUMMARY,
        StreamEventType.PROGRESS,
        StreamEventType.SOURCE,
    }:
        return build_chunks("reasoning", f"- {event.content}\n\n")
    return []


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
