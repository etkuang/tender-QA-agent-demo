# coding: utf-8
# @Author: Wang Qingkang

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["assistant", "user"]
    content: str


class AgentStreamRequest(BaseModel):
    user_message: str
    history_messages: list[Message] = Field(default_factory=list)


class AgentTransportChunk(BaseModel):
    type: Literal["assistant", "reasoning", "error"]
    content: str
