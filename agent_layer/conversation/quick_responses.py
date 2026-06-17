# coding: utf-8
# @Author: Wang Qingkang

DEFAULT_EMPTY_RESPONSE = "您好，请输入具体问题。"
DEFAULT_QUICK_RESPONSE = "您好，请问有什么可以帮助您？"

QUICK_RESPONSES = {
    "你好": "您好！我是招投标六类智能问答助手，请问有什么可以帮您？",
    "您好": "您好！请问您想查询政策、招标、舆情、企业、价格还是商品信息？",
    "hi": "Hello！请问有什么可以帮您？",
    "hello": "Hello！请问有什么可以帮您？",
    "嗨": "您好！请问有什么可以帮您？",
    "在吗": "在的，请问有什么可以帮您？",
    "在不在": "在的，请问有什么可以帮您？",
    "有人吗": "在的，请问有什么可以帮您？",
    "谢谢": "不客气，有问题随时问我。",
    "感谢": "不客气。",
    "thanks": "You're welcome. Feel free to ask me any question.",
    "thank": "You're welcome. Feel free to ask me any question.",
    "再见": "再见！如有问题，随时回来咨询。",
    "拜拜": "再见！如有问题，随时回来咨询。",
    "bye": "Goodbye! Feel free to come back if you have more questions.",
    "goodbye": "Goodbye! Feel free to come back if you have more questions.",
}

GREETING_KEYWORDS = ["你好", "您好", "hi", "hello", "嗨", "在吗", "在不在", "有人吗"]
THANKS_KEYWORDS = ["谢谢", "感谢", "thanks", "thank"]
GOODBYE_KEYWORDS = ["再见", "拜拜", "bye", "goodbye"]

QUICK_RESPONSE_KEYWORD_GROUPS = (
    GREETING_KEYWORDS,
    THANKS_KEYWORDS,
    GOODBYE_KEYWORDS,
)