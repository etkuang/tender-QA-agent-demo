# coding: utf-8

import json

from langchain_core.language_models import BaseChatModel

from common.logger import get_logger
from agent_layer.classification.prompts import QUESTION_DECOMPOSITION_PROMPT
from agent_layer.config import Settings
from agent_layer.errors import ClassificationError, raise_model_error
from agent_layer.schemas import ChildTask, ChildTaskList

logger = get_logger("agent.classification")


class QuestionClassifier:
    def __init__(self, model: BaseChatModel, settings: Settings):
        self.settings = settings
        structured_model = model.with_structured_output(
            ChildTaskList,
            method="json_mode",
        )
        self.chain = QUESTION_DECOMPOSITION_PROMPT | structured_model

    async def classify(self, question: str, history: str = "") -> list[ChildTask]:
        try:
            result = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "question": question,
                    "history": history,
                    "schema": json.dumps(ChildTaskList.model_json_schema(), ensure_ascii=False),
                }
            )
        except Exception as exc:
            logger.exception("structured question understanding failed")
            raise_model_error(exc, ClassificationError)
        if not isinstance(result, ChildTaskList):
            logger.error("question understanding returned unexpected type | type=%s", type(result).__name__)
            raise ClassificationError
        return result.root
