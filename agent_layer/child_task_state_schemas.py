# coding: utf-8
# @Author: Wang Qingkang

from pydantic import BaseModel, Field

from agent_layer.structured_output_schemas import AnswerStatus
from common.api_contracts.agent_api import Message


class PrerequisiteAnswer(BaseModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ChildTaskInputState(BaseModel):
    session_id: str
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requires_fresh_data: bool = False
    history_messages: list[Message] = Field(default_factory=list)
    prerequisite_answers: list[PrerequisiteAnswer] = Field(
        default_factory=list
    )


class ChildTaskOutput(BaseModel):
    task_id: str = Field(min_length=1)
    status: AnswerStatus
    answer: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ChildTaskOutputState(BaseModel):
    child_task_output: ChildTaskOutput


class ChildTaskState(ChildTaskInputState):
    child_task_output: ChildTaskOutput | None = None