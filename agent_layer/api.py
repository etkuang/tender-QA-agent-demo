# coding: utf-8
# @Author: Wang Qingkang

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from common.logger import get_logger, reset_request_id, set_request_id
from agent_layer.agent import TenderAgentRuntime
from agent_layer.config import settings
from agent_layer.schemas import AgentStreamChunk, AgentStreamRequest

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
)
runtime = TenderAgentRuntime()
logger = get_logger("agent.api")


@app.post("/chat/stream")
async def stream_chat(req: AgentStreamRequest, request: Request):
    request_id = request.headers.get("X-Request-ID")
    if request_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Request-ID header")

    token = set_request_id(request_id)
    try:
        logger.info("agent.stream start | history_messages=%s", len(req.history_messages))
        try:
            result = await runtime.ask(req.user_message, history_messages=req.history_messages)
        except Exception:
            logger.exception("agent.stream failed")
            raise HTTPException(status_code=500, detail="Agent inference failed.")

        logger.info(
            "agent.stream finished | route=%s | processing_time=%.3f | sources=%s",
            result.route,
            result.processing_time,
            len(result.sources),
        )

        async def ndjson_stream():
            for index in range(0, len(result.answer), settings.stream_chunk_size):
                content = result.answer[index:index + settings.stream_chunk_size]
                payload = AgentStreamChunk(type="assistant", content=content)
                yield f"{json.dumps(payload.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
    finally:
        reset_request_id(token)
