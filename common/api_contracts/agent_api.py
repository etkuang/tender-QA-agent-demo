# coding: utf-8
# @Author: Wang Qingkang

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str


class SessionContext(BaseModel):
    entity_state: dict[str, Any] = Field(default_factory=dict)
    conversation_summary: str = ""
    trusted_attributes: dict[str, Any] = Field(default_factory=dict, exclude=True)


class GenerationOptions(BaseModel):
    include_progress: bool = True


class AgentStreamRequest(BaseModel):
    user_message: str
    history_messages: list[Message] = Field(default_factory=list)
    session_context: SessionContext = Field(default_factory=SessionContext)
    generation_options: GenerationOptions = Field(default_factory=GenerationOptions)


class AgentTransportChunk(BaseModel):
    type: Literal["assistant", "reasoning", "error"]
    content: str
    code: str | None = None
    error_id: str | None = None