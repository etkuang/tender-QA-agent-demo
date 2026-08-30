# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer_old.config import Settings
from agent_layer_old.schemas import QuickResponseDecision, QuickResponseType

logger = get_logger("agent.quick_response")


QUICK_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个招投标 AI 助手的快捷回复类别分类器。
判断当前用户消息是否只属于可以固定话术回复的类别。

只输出 JSON，不要输出解释或多余文本。
格式示例：
{"quick_response_type": "none"}

quick_response_type 只能取以下值之一：
- greeting_zh：中文问候、开场寒暄、询问是否在线。
- greeting_en：英文问候、开场寒暄、询问是否在线。
- thanks_zh：中文感谢。
- thanks_en：英文感谢。
- goodbye_zh：中文告别、结束对话。
- goodbye_en：英文告别、结束对话。
- capabilities_zh：中文询问 AI 助手能力、适用范围、能做什么。
- capabilities_en：英文询问 AI 助手能力、适用范围、能做什么。
- wellbeing_zh：中文询问助手状态、是否正常、过得如何。
- wellbeing_en：英文询问助手状态、是否正常、过得如何。
- acknowledgement_zh：中文单纯确认、收到、好的、明白。
- acknowledgement_en：英文单纯确认、收到、好的、明白。
- compliment_zh：中文表扬、认可助手。
- compliment_en：英文表扬、认可助手。
- apology_zh：中文道歉或表示刚才说错。
- apology_en：英文道歉或表示刚才说错。
- none：不应使用快捷回复，进入正常业务工作流。

判断规则：
- 用户消息只表达问候、谢意、告别、询问 AI 能力、询问 AI 状态、确认收到、表扬或道歉时，选择对应类别。
- 用户消息包含招投标、政策、项目、企业、价格、商品、舆情或其他可回答问题时，选择 none。
- 用户消息是在补充条件、纠错、继续追问、回答是/否或反馈上一轮业务答案时，选择 none。
- “好的、明白、收到、OK、understood”等只有在没有业务问题或补充条件时才选择 acknowledgement。
- “你能做什么、what can you do、你的功能是什么”等选择 capabilities。
- 语言按用户消息主要语言选择 zh 或 en；中英文混合时选择更适合直接回复的语言。
- 无法判断时选择 none。""",
        ),
        (
            "human",
            "用户最新消息：\n{question}",
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
                }
            )
        except Exception:
            logger.exception("quick response classification failed; using normal workflow")
            return QuickResponseDecision(quick_response_type=QuickResponseType.NONE)
        return result
