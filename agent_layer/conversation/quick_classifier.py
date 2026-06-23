# coding: utf-8
# @Author: Wang Qingkang

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import ClassificationError, raise_model_error
from agent_layer.schemas import QuickResponseDecision

logger = get_logger("agent.quick_response")


QUICK_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是招投标智能问答系统的快捷回复意图分类器。只判断用户最新消息是否应该由固定话术直接回复，不回答业务问题。

可选 intent：
- greeting_zh：中文问候、开场寒暄、询问是否在线。
- greeting_en：英文问候、开场寒暄、询问是否在线。
- thanks_zh：中文感谢。
- thanks_en：英文感谢。
- goodbye_zh：中文告别、结束对话。
- goodbye_en：英文告别、结束对话。
- capabilities_zh：中文询问系统能力、适用范围、能做什么。
- capabilities_en：英文询问系统能力、适用范围、能做什么。
- wellbeing_zh：中文询问助手状态、是否正常、过得如何。
- wellbeing_en：英文询问助手状态、是否正常、过得如何。
- acknowledgement_zh：中文单纯确认、收到、好的、明白。
- acknowledgement_en：英文单纯确认、收到、好的、明白。
- compliment_zh：中文表扬、认可助手。
- compliment_en：英文表扬、认可助手。
- apology_zh：中文道歉或表示刚才说错。
- apology_en：英文道歉或表示刚才说错。
- none：不应使用快捷回复，必须进入正常业务工作流。

分类规则：
- 只有当用户最新消息的主要意图是纯社交或会话控制时，才选择快捷回复 intent。
- 如果消息包含任何招投标、政策、项目、企业、价格、商品、舆情或其他可回答问题，即使开头或结尾有寒暄，也必须选择 none。
- 如果消息包含补充条件、纠错、继续、是/否、上一轮追问或对业务答案的反馈，选择 none。
- “好的、明白、收到、OK、understood”等只有在没有业务问题或补充条件时才选择 acknowledgement。
- “你能做什么、what can you do、你的功能是什么”等选择 capabilities。
- 语言按用户消息主要语言选择 zh 或 en；中英文混合时选择更适合直接回复的语言。
- 不确定时选择 none。""",
        ),
        (
            "human",
            "用户最新消息：\n{question}\n\nJSON Schema：\n{schema}",
        ),
    ]
)


class QuickResponseClassifier:
    def __init__(self, model: BaseChatModel, settings: Settings):
        self.settings = settings
        structured_model = model.with_structured_output(
            QuickResponseDecision,
            method="json_mode",
        )
        self.chain = QUICK_RESPONSE_PROMPT | structured_model

    async def classify(self, question: str) -> QuickResponseDecision:
        try:
            result = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "question": question,
                    "schema": json.dumps(QuickResponseDecision.model_json_schema(), ensure_ascii=False),
                }
            )
        except Exception as exc:
            logger.exception("quick response classification failed")
            raise_model_error(exc, ClassificationError)
        if not isinstance(result, QuickResponseDecision):
            logger.error("quick response classification returned unexpected type | type=%s", type(result).__name__)
            raise ClassificationError
        return result