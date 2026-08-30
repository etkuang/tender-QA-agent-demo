# coding: utf-8
# @Author: Wang Qingkang

from pydantic import BaseModel, ConfigDict, Field

from agent_layer.structured_output_schemas import ChildTask
from common.api_contracts.agent_api import Message


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentInputState(StateModel):
    user_message: str
    history_messages: list[Message] = Field(default_factory=list)


class AgentState(AgentInputState):
    child_tasks: list[ChildTask] = Field(default_factory=list)
    answer: str = ""


class AgentOutputState(StateModel):
    answer: str
