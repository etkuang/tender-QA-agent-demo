# coding: utf-8

from typing import TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(extra="allow")


class OpenAIChatRequest(BaseModel):
    model: str = "tender-agent"
    messages: list[OpenAIMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    stream: bool = False
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


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
    choices: list[OpenAIChoice]


class IntentDecision(BaseModel):
    intent_type: str = Field(default="other")
    complexity: str = Field(default="single_step")


class AskResult(BaseModel):
    answer: str
    route: str
    processing_time: float
    sources: list[dict] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    rewritten_question: str
    history_context: str
    messages: list[dict]
    intent: IntentDecision
    route: str
    documents: list[Document]
    context: str
    answer: str
    sources: list[dict]
    processing_time: float
