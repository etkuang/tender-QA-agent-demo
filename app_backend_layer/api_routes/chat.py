# coding: utf-8
# @Author: Wang Qingkang

import json

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app_backend_layer.config import settings
from common.logger import get_logger, get_request_id, reset_request_id, set_request_id
from app_backend_layer.history.history_db import HistoryManager
from app_backend_layer.schemas import ChatRequest, StreamChunk

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger("backend.chat")


def get_db(request: Request) -> HistoryManager:
    return request.app.state.db


def stream_error_handler(err: Exception, request_id: str) -> tuple[str, str]:
    if isinstance(err, httpx.HTTPStatusError):
        if err.response.status_code == 429:
            return f"请求过于频繁，请稍后重试。错误编号：{request_id}", "agent_rate_limited"
        if err.response.status_code in {502, 503, 504}:
            return f"Agent 服务暂时不可用，请稍后重试。错误编号：{request_id}", "agent_service_unavailable"
    if isinstance(err, httpx.TimeoutException):
        return f"Agent 响应超时，请稍后重试。错误编号：{request_id}", "agent_timeout"
    if isinstance(err, httpx.NetworkError):
        return f"无法连接 Agent 服务，请稍后重试。错误编号：{request_id}", "agent_network_error"
    return f"本次回答生成失败，系统已记录错误日志。错误编号：{request_id}", "internal_stream_error"


@router.post("/stream")
async def stream_chat(req: ChatRequest, db: HistoryManager = Depends(get_db)):
    request_id = get_request_id()
    logger.info("chat.stream start | session_id=%s", req.session_id)

    history_messages = await db.load_messages(req.session_id)
    agent_payload = {
        "user_message": req.user_message,
        "history_messages": [
            {"role": message["type"], "content": message["content"]}
            for message in history_messages
            if message["type"] in {"user", "assistant"}
        ],
    }

    async def generate_and_save():
        token = set_request_id(request_id)
        full_response = ""
        full_reasoning = ""
        full_error = ""
        try:
            await db.append_message(req.session_id, req.session_title, "user", req.user_message)

            async with httpx.AsyncClient(timeout=settings.STREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{settings.AGENT_BASE_URL}chat/stream",
                    json=agent_payload,
                    headers={"X-Request-ID": request_id},
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        payload = StreamChunk.model_validate_json(line)
                        if payload.type == "reasoning":
                            full_reasoning += payload.content
                        elif payload.type == "assistant":
                            full_response += payload.content
                        elif payload.type == "error":
                            full_error += payload.content
                        yield f"{json.dumps(payload.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

            if full_reasoning:
                await db.append_message(req.session_id, req.session_title, "reasoning", full_reasoning)
            if full_response:
                await db.append_message(req.session_id, req.session_title, "assistant", full_response)
            if full_error:
                await db.append_message(req.session_id, req.session_title, "error", full_error)
            logger.info(
                "chat.stream finished | session_id=%s | response_chars=%s",
                req.session_id,
                len(full_response),
            )
        except Exception as stream_err:
            content, code = stream_error_handler(stream_err, request_id)
            logger.exception(
                "chat.stream failed | session_id=%s | request_id=%s | code=%s",
                req.session_id,
                request_id,
                code,
            )
            await db.append_message(req.session_id, req.session_title, "error", content)
            error_chunk = StreamChunk(type="error", content=content, code=code, error_id=request_id)
            yield f"{json.dumps(error_chunk.model_dump(), ensure_ascii=False)}\n".encode("utf-8")
        finally:
            reset_request_id(token)

    return StreamingResponse(generate_and_save(), media_type="application/x-ndjson")
