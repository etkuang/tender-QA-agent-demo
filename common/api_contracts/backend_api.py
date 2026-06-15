# coding: utf-8
# @Author: Wang Qingkang

from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    session_title: str
    user_message: str


class StreamChunk(BaseModel):
    type: Literal["assistant", "reasoning", "error"]
    content: str
    code: str | None = None
    error_id: str | None = None


class SessionMeta(BaseModel):
    session_id: str
    title: str


class StoredMessage(BaseModel):
    type: Literal["user", "assistant", "reasoning", "error"]
    content: str
