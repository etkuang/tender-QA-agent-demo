# coding: utf-8

import json

from langchain_core.language_models import BaseChatModel

from common.logger import get_logger
from agent_layer.classification.prompts import CLASSIFICATION_PROMPT
from agent_layer.config import Settings
from agent_layer.errors import ClassificationError, raise_model_error
from agent_layer.schemas import QuestionClassification

logger = get_logger("agent.classification")


class QuestionClassifier:
    def __init__(self, model: BaseChatModel, settings: Settings):
        self.settings = settings
        structured_model = model.with_structured_output(
            QuestionClassification,
            method=settings.structured_output_method,
        )
        self.chain = CLASSIFICATION_PROMPT | structured_model

    async def classify(self, question: str, context_summary: str = "") -> QuestionClassification:
        try:
            result = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "question": question,
                    "context_summary": context_summary,
                    "schema": json.dumps(QuestionClassification.model_json_schema(), ensure_ascii=False),
                }
            )
        except Exception as exc:
            logger.exception("structured classification failed")
            raise_model_error(exc, ClassificationError)
        if not isinstance(result, QuestionClassification):
            logger.error("classification returned unexpected type | type=%s", type(result).__name__)
            raise ClassificationError
        return result
