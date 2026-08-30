# coding: utf-8
# @Author: Wang Qingkang

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent_layer.bootstrap import build_application
from common.api_contracts.agent_api import AgentStreamRequest, AgentTransportChunk
from common.logger import get_logger, reset_request_id, set_request_id

logger = get_logger("agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.application = await build_application()
    try:
        yield
    finally:
        await app.state.application.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/stream")
async def stream_chat(req: AgentStreamRequest, request: Request) -> StreamingResponse:
    request_id = request.headers.get("X-Request-ID")
    if request_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Request-ID header")

    async def ndjson_stream() -> AsyncIterator[bytes]:
        token = set_request_id(request_id)
        try:
            logger.info("agent.stream start | history_messages=%s", len(req.history_messages))
            async for chunk in request.app.state.application.stream(
                    req.user_message,
                    req.history_messages,
            ):
                if await request.is_disconnected():
                    logger.info("agent.stream cancelled by client")
                    break
                payload = chunk.model_dump(mode="json", exclude_none=True)
                yield f"{json.dumps(payload, ensure_ascii=False)}\n".encode("utf-8")
            logger.info("agent.stream finished")
        finally:
            reset_request_id(token)

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")