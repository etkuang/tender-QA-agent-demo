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
    type: Literal["thinking", "text"]
    content: str


class SessionMeta(BaseModel):
    session_id: str
    title: str


class StoredMessage(BaseModel):
    type: str
    data: dict


class DeleteSessionResponse(BaseModel):
    ok: bool
    session_id: str
