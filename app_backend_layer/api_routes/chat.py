# coding: utf-8
# @Author: Wang Qingkang

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, messages_from_dict
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from app_backend_layer.core.config import settings
from common.logging import get_logger, get_request_id
from app_backend_layer.history.history_db import HistoryManager
from app_backend_layer.models.schemas import ChatRequest, StreamChunk

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger("backend.chat")


def extract_text_from_chunk(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            text_value = item.get("text")
            if isinstance(text_value, str):
                parts.append(text_value)
    return "".join(parts)


def get_db(request: Request) -> HistoryManager:
    return request.app.state.db


def is_retryable_stream_error(err: Exception) -> bool:
    return isinstance(err, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError))


@router.post("/stream")
async def stream_chat(req: ChatRequest, db: HistoryManager = Depends(get_db)):
    try:
        agent_base_url = f"{settings.AGENT_BASE_URL}".rstrip("/")
        logger.info("chat.stream start | session_id=%s", req.session_id)

        context_window = []
        history_messages = await db.load_messages(req.session_id, with_reasoning=False)
        context_window.extend(messages_from_dict(history_messages))
        context_window.append(HumanMessage(content=req.prompt))

        llm = ChatOpenAI(
            model=settings.AGENT_MODEL_NAME,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            api_key=settings.AGENT_API_KEY,
            base_url=agent_base_url,
            timeout=settings.STREAM_TIMEOUT,
            streaming=True,
            extra_body={"metadata": {"session_id": req.session_id, "request_id": get_request_id()}},
        )

        async def generate_and_save():
            full_response = ""
            try:
                await db.append_message(req.session_id, req.session_title, "human", req.prompt)

                async for chunk in llm.astream(context_window):
                    content = extract_text_from_chunk(chunk)
                    if not content:
                        continue

                    full_response += content
                    payload = StreamChunk(type="text", content=content)
                    yield f"{json.dumps(payload.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

                await db.append_message(req.session_id, req.session_title, "ai", full_response)
                logger.info("chat.stream finished | session_id=%s | response_chars=%s", req.session_id, len(full_response))
            except Exception as stream_err:
                error_id = uuid.uuid4().hex[:8]
                logger.exception("chat.stream failed | session_id=%s | error_id=%s", req.session_id, error_id)
                if is_retryable_stream_error(stream_err):
                    content = f"服务暂时不可用，请稍后重试。错误编号：{error_id}"
                    code = "retryable_stream_error"
                else:
                    content = f"本次回答生成失败，系统已记录错误日志。错误编号：{error_id}"
                    code = "internal_stream_error"
                error_chunk = StreamChunk(type="error", content=content, code=code, error_id=error_id)
                yield f"{json.dumps(error_chunk.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

        return StreamingResponse(generate_and_save(), media_type="application/x-ndjson")
    except Exception:
        error_id = uuid.uuid4().hex[:8]
        logger.exception("chat.stream init failed | session_id=%s | error_id=%s", req.session_id, error_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Inference initialization pipeline crashed.", "error_id": error_id},
        )
