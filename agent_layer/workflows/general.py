# coding: utf-8

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer.errors import GenerationError, raise_model_error

logger = get_logger("agent.workflows.general")


GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是通用中文助手。直接回答通用知识问题，不调用招投标专业工具。"
            "不要声称查询了数据库、知识库或互联网。",
        ),
        ("human", "问题：\n{question}\n\n相关对话摘要：\n{history}"),
    ]
)


class GeneralWorkflow:
    def __init__(self, model: BaseChatModel):
        self.chain = GENERAL_PROMPT | model | StrOutputParser()

    async def run(self, question: str, history: str) -> str:
        try:
            return await self.chain.ainvoke({"question": question, "history": history})
        except Exception as exc:
            logger.exception("general answer generation failed")
            raise_model_error(exc, GenerationError)
