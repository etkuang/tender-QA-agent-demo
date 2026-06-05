# coding: utf-8
# @Author: Wang Qingkang

from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    session_title: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int


class StreamChunk(BaseModel):
    type: Literal["thinking", "text", "error"]
    content: str
    code: str | None = None
    error_id: str | None = None


class SessionMeta(BaseModel):
    session_id: str
    title: str


class StoredMessage(BaseModel):
    type: str
    data: dict


class DeleteSessionResponse(BaseModel):
    ok: bool
    session_id: str
