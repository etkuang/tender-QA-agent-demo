# coding: utf-8

from langchain_core.language_models import BaseChatModel

from common.logger import get_logger
from agent_layer_old.question_decomposition.prompts import QUESTION_DECOMPOSITION_PROMPT
from agent_layer_old.config import Settings
from agent_layer_old.errors import DecompositionError, raise_model_error
from agent_layer_old.schemas import ChildTask, ChildTaskList

logger = get_logger("agent.decomposition")


class QuestionDecomposer:
    def __init__(self, model: BaseChatModel, settings: Settings):
        self.settings = settings
        structured_model = model.with_structured_output(
            ChildTaskList,
            method="json_mode",
        )
        self.chain = QUESTION_DECOMPOSITION_PROMPT | structured_model

    async def decompose(self, question: str, history: str = "") -> list[ChildTask]:
        try:
            result = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "question": question,
                    "history": history,
                }
            )
        except Exception as exc:
            logger.exception("structured question decomposition failed")
            raise_model_error(exc, DecompositionError)
        return result.root