# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.structured_output_schemas import QuickResponseType


EMPTY_QUICK_RESPONSE = "您好，请输入具体问题。"

QUICK_RESPONSES = {
    QuickResponseType.GREETING_ZH: {
        "intent_label": "中文问候",
        "content": "您好！我是招投标智能问答助手。我可以帮助您了解政策法规，并查询和分析招标项目、舆情动态、企业信息和商品资料。请告诉我您想了解的问题。",
    },
    QuickResponseType.GREETING_EN: {
        "intent_label": "英文问候",
        "content": "Hello! I’m a tendering and procurement Q&A assistant. I can help with policies and regulations, tenders, public opinion, companies, and products. What would you like to know?",
    },
    QuickResponseType.THANKS_ZH: {
        "intent_label": "中文感谢",
        "content": "不客气。如有其他招投标相关问题，欢迎继续提问。",
    },
    QuickResponseType.THANKS_EN: {
        "intent_label": "英文感谢",
        "content": "You’re welcome. Feel free to ask another tendering or procurement-related question.",
    },
    QuickResponseType.GOODBYE_ZH: {
        "intent_label": "中文告别",
        "content": "再见！如有招投标相关问题，欢迎随时回来咨询。",
    },
    QuickResponseType.GOODBYE_EN: {
        "intent_label": "英文告别",
        "content": "Goodbye! You’re welcome to return whenever you have another tendering or procurement question.",
    },
    QuickResponseType.CAPABILITIES_ZH: {
        "intent_label": "中文询问助手能力",
        "content": "我是招投标智能问答助手，可以协助您了解政策法规，并查询和分析招标项目、舆情动态、企业信息和商品资料。您可以直接描述查询对象、地区、时间范围和关注的问题。",
    },
    QuickResponseType.CAPABILITIES_EN: {
        "intent_label": "英文询问助手能力",
        "content": "I’m a tendering and procurement Q&A assistant. I can help with policies, tenders, public opinion, companies, and products. You can specify the subject, region, time range, and information you need.",
    },
    QuickResponseType.WELLBEING_ZH: {
        "intent_label": "中文询问助手状态",
        "content": "谢谢关心，我运行正常，可以随时为您处理招投标相关问题。",
    },
    QuickResponseType.WELLBEING_EN: {
        "intent_label": "英文询问助手状态",
        "content": "Thank you for asking. I’m operating normally and ready to help with tendering and procurement questions.",
    },
    QuickResponseType.ACKNOWLEDGEMENT_ZH: {
        "intent_label": "中文确认",
        "content": "好的。需要继续查询时，请直接告诉我具体问题。",
    },
    QuickResponseType.ACKNOWLEDGEMENT_EN: {
        "intent_label": "英文确认",
        "content": "Understood. When you are ready, tell me what you would like to investigate.",
    },
    QuickResponseType.COMPLIMENT_ZH: {
        "intent_label": "中文表扬",
        "content": "谢谢您的认可。我会继续尽力提供准确、清晰的回答。",
    },
    QuickResponseType.COMPLIMENT_EN: {
        "intent_label": "英文表扬",
        "content": "Thank you. I’ll continue working to provide accurate and clear answers.",
    },
    QuickResponseType.APOLOGY_ZH: {
        "intent_label": "中文道歉",
        "content": "没关系。您可以继续说明问题或补充需要查询的信息。",
    },
    QuickResponseType.APOLOGY_EN: {
        "intent_label": "英文道歉",
        "content": "No problem. You can continue explaining the question or provide additional details.",
    },
}