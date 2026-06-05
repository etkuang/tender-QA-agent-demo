# coding: utf-8

import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from common.logging import get_logger, reset_request_id, set_request_id
from agent_layer.agent import TenderAgentRuntime
from agent_layer.config import settings
from agent_layer.state import (AskRequest, AskResponse, OpenAIChatRequest, OpenAIChatResponse, OpenAIChoice,
                               OpenAIChoiceMessage)

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
)
runtime = TenderAgentRuntime()
logger = get_logger("agent.api")


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
    request_id = uuid.uuid4().hex
    token = set_request_id(request_id)
    try:
        logger.info("agent.ask start | session_id=%s", session_id)
        result = await runtime.ask(req.question, session_id=session_id)
        logger.info(
            "agent.ask finished | session_id=%s | route=%s | processing_time=%.3f | sources=%s",
            session_id,
            result.route,
            result.processing_time,
            len(result.sources),
        )
        return AskResponse(
            answer=result.answer,
            processing_time=result.processing_time,
            session_id=session_id,
            route=result.route,
            sources=result.sources,
        )
    except Exception:
        logger.exception("agent.ask failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Agent inference failed.")
    finally:
        reset_request_id(token)


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "tender-agent", "object": "model", "owned_by": "tender-team"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest):
    request_id = req.metadata.get("request_id") or uuid.uuid4().hex
    token = set_request_id(request_id)
    try:
        user_messages = [message for message in req.messages if message.role == "user" and message.content]
        if not user_messages:
            logger.warning("agent.chat invalid request | reason=no_user_message")
            raise HTTPException(status_code=400, detail="No user message found in request.messages")

        message_payload = [message.model_dump() for message in req.messages]
        session_id = req.metadata.get("session_id", "")
        logger.info("agent.chat start | session_id=%s | stream=%s | messages=%s", session_id, req.stream, len(req.messages))
        try:
            result = await runtime.ask(
                user_messages[-1].content,
                session_id=session_id,
                messages=message_payload,
            )
        except Exception:
            logger.exception("agent.chat failed | session_id=%s", session_id)
            raise HTTPException(status_code=500, detail="Agent inference failed.")

        logger.info(
            "agent.chat finished | session_id=%s | route=%s | processing_time=%.3f | sources=%s",
            session_id,
            result.route,
            result.processing_time,
            len(result.sources),
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if not req.stream:
            return OpenAIChatResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIChoiceMessage(role="assistant", content=result.answer),
                        finish_reason="stop",
                    )
                ],
            )

        async def sse_stream():
            head = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"

            for index in range(0, len(result.answer), settings.stream_chunk_size):
                body = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": result.answer[index:index + settings.stream_chunk_size]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

            tail = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")
    finally:
        reset_request_id(token)
