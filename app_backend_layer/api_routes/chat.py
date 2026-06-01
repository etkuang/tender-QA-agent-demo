# coding: utf-8
# @Author: Wang Qingkang

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, messages_from_dict
from langchain_openai import ChatOpenAI

from app_backend_layer.core.config import settings
from app_backend_layer.core.logger import get_logger
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
        )

        async def generate_and_save():
            full_response = ""
            full_reasoning = ""
            try:
                await db.append_message(req.session_id, req.session_title, "human", req.prompt)

                bootstrap = StreamChunk(type="thinking", content="Analyzing context and preparing answer...")
                full_reasoning += bootstrap.content
                yield f"{json.dumps(bootstrap.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

                async for chunk in llm.astream(context_window):
                    content = extract_text_from_chunk(chunk)
                    if not content:
                        continue

                    full_response += content
                    payload = StreamChunk(type="text", content=content)
                    yield f"{json.dumps(payload.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

                if full_reasoning:
                    await db.append_message(req.session_id, req.session_title, "reasoning", full_reasoning)
                await db.append_message(req.session_id, req.session_title, "ai", full_response)
            except Exception as stream_err:
                logger.exception("chat.stream failed | session_id=%s", req.session_id)
                error_chunk = StreamChunk(
                    type="text",
                    content=f"\n\n[Streaming Pipeline Error: {type(stream_err).__name__}: {stream_err}]",
                )
                yield f"{json.dumps(error_chunk.model_dump(), ensure_ascii=False)}\n".encode("utf-8")

        return StreamingResponse(generate_and_save(), media_type="application/x-ndjson")
    except Exception as err:
        logger.exception("chat.stream init failed | session_id=%s", req.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference initialization pipeline crashed: {type(err).__name__}: {err}",
        )
