# coding: utf-8

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from common.logger import get_logger, reset_request_id, set_request_id
from agent_layer.bootstrap import build_application
from agent_layer.config import settings
from agent_layer.schemas import AgentStreamRequest

logger = get_logger("agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.application = build_application()
    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)


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
                req.session_context,
                req.generation_options,
            ):
                if await request.is_disconnected():
                    logger.info("agent.stream cancelled by client")
                    break
                payload = event.model_dump(mode="json", exclude_none=True)
                yield f"{json.dumps(payload, ensure_ascii=False)}\n".encode("utf-8")
            logger.info("agent.stream finished")
        finally:
            reset_request_id(token)

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
