# coding: utf-8
# @Author: Wang Qingkang

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class QuickResponseType(StrEnum):
    GREETING_ZH = "greeting_zh"
    GREETING_EN = "greeting_en"
    THANKS_ZH = "thanks_zh"
    THANKS_EN = "thanks_en"
    GOODBYE_ZH = "goodbye_zh"
    GOODBYE_EN = "goodbye_en"
    CAPABILITIES_ZH = "capabilities_zh"
    CAPABILITIES_EN = "capabilities_en"
    WELLBEING_ZH = "wellbeing_zh"
    WELLBEING_EN = "wellbeing_en"
    ACKNOWLEDGEMENT_ZH = "acknowledgement_zh"
    ACKNOWLEDGEMENT_EN = "acknowledgement_en"
    COMPLIMENT_ZH = "compliment_zh"
    COMPLIMENT_EN = "compliment_en"
    APOLOGY_ZH = "apology_zh"
    APOLOGY_EN = "apology_en"
    NONE = "none"


class QuickResponseDecision(BaseModel):
    quick_response_type: QuickResponseType


class Category(StrEnum):
    POLICY = "policy"
    TENDER = "tender"
    PUBLIC_OPINION = "public_opinion"
    COMPANY = "company"
    PRODUCT = "product"
    OTHER = "other"
    UNCLEAR = "unclear"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_CLARIFICATION = "needs_clarification"


class AnswerDraft(BaseModel):
    status: AnswerStatus
    answer: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ChildTask(BaseModel):
    task_id: str
    question: str
    category: Category
    depends_on: list[str] = Field(default_factory=list)
    requires_fresh_data: bool = False
    clarification_question: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_task_contract(self) -> Self:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("depends_on must not contain duplicated task ids")
        has_clarification = self.clarification_question is not None
        if (self.category == Category.UNCLEAR) != has_clarification:
            raise ValueError("Only unclear child tasks must provide clarification_question")
        return self


class ChildTaskList(BaseModel):
    child_tasks: list[ChildTask] = Field(min_length=1)