# coding: utf-8
# @Author: Wang Qingkang

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent_layer.application.services.ask_service import AskService
from agent_layer.interfaces.http.schemas import (
    AskRequest,
    AskResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIChoiceMessage,
)

router = APIRouter()


def bind_routes(ask_service: AskService) -> APIRouter:
    @router.post("/api/v1/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
        answer, route, elapsed, sources = await ask_service.ask(req.question)
        return AskResponse(
            answer=answer,
            processing_time=elapsed,
            session_id=session_id,
            route=route,
            sources=sources,
        )

    @router.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": "tender-agent", "object": "model", "owned_by": "tender-team"}]}

    @router.post("/v1/chat/completions")
    async def chat_completions(req: OpenAIChatRequest):
        user_messages = [m for m in req.messages if m.role == "user" and m.content]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in request.messages")

        answer, _, _, _ = await ask_service.ask(user_messages[-1].content)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if not req.stream:
            return OpenAIChatResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[OpenAIChoice(index=0, message=OpenAIChoiceMessage(role="assistant", content=answer), finish_reason="stop")],
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

            step = 32
            for i in range(0, len(answer), step):
                body = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": answer[i:i + step]}, "finish_reason": None}],
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

    return router