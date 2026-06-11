# coding: utf-8
# @Author: Wang Qingkang

from typing import Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class AgentStreamRequest(BaseModel):
    user_message: str
    history_messages: list[dict] = Field(default_factory=list)


class AgentStreamChunk(BaseModel):
    type: Literal["assistant", "reasoning", "error"]
    content: str
    code: str | None = None
    error_id: str | None = None


class IntentDecision(BaseModel):
    intent_type: str = Field(default="other")
    complexity: str = Field(default="single_step")


class AskResult(BaseModel):
    answer: str
    route: str
    processing_time: float
    sources: list[dict] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    user_message: str
    rewritten_user_message: str
    history_context: str
    history_messages: list[dict]
    intent: IntentDecision
    route: str
    documents: list[Document]
    context: str
    answer: str
    sources: list[dict]
    processing_time: float
