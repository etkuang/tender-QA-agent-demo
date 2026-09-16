# coding: utf-8
# @Author: Wang Qingkang


from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_layer.evidence_schemas import WebsiteEvidenceDraft

INTERNET_SEARCH_TOOL_NAME = "internet_search"

INTERNET_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": INTERNET_SEARCH_TOOL_NAME,
        "description": "检索与问题相关的公开互联网资料。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "聚焦检索目标的查询语句。",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class InternetSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class InternetSearchStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class InternetSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InternetSearchStatus
    evidence: list[WebsiteEvidenceDraft] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        if self.status == InternetSearchStatus.COMPLETED:
            if self.reason:
                raise ValueError(
                    "Completed Internet searches must not include a reason"
                )
        elif self.evidence or not self.reason:
            raise ValueError(
                "Failed Internet searches require a reason and no evidence"
            )
        return self


class InternetSearchGateway(Protocol):
    async def search(
        self,
        query: str,
        limit: int,
    ) -> InternetSearchResult:
        """Return at most limit evidence drafts without IDs or session metadata.

        The evidence-generating model may split one source into multiple drafts.
        Each draft's content must be source-faithful and self-contained without its title.
        The evidence pool assigns session ownership and receives IDs from Milvus.
        """