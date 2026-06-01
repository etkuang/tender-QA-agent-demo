# coding: utf-8
# @Author: Wang Qingkang

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    processing_time: float
    session_id: str
    route: str
    sources: list[dict] = Field(default_factory=list)


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "tender-agent"
    messages: List[OpenAIMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    stream: bool = False


class OpenAIChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChoiceMessage
    finish_reason: str = "stop"


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]