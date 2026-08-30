# coding: utf-8
# @Author: Wang Qingkang

from agent_layer_old.schemas import QuickResponseType


EMPTY_QUICK_RESPONSE = {
    "kind": "empty",
    "route_content": "当前输入为空，我会提示用户补充具体问题。",
    "content": "您好，请输入具体问题。",
}

QUICK_RESPONSES = {
    QuickResponseType.GREETING_ZH: {
        "kind": QuickResponseType.GREETING_ZH.value,
        "route_content": "用户在进行问候，我会直接回应并简要说明可处理的问题范围。",
        "content": "您好！我是招投标智能问答助手。我可以帮助您了解政策法规，并查询和分析招标项目、舆情动态、企业信息和商品资料。请告诉我您想了解的问题。",
    },
    QuickResponseType.GREETING_EN: {
        "kind": QuickResponseType.GREETING_EN.value,
        "route_content": "用户在进行英文问候，我会直接回应并简要说明可处理的问题范围。",
        "content": "Hello! I’m a tendering and procurement Q&A assistant. I can help with policies and regulations, tenders, public opinion, companies, and products. What would you like to know?",
    },
    QuickResponseType.THANKS_ZH: {
        "kind": QuickResponseType.THANKS_ZH.value,
        "route_content": "用户在表达感谢，我会直接回应。",
        "content": "不客气。如有其他招投标相关问题，欢迎继续提问。",
    },
    QuickResponseType.THANKS_EN: {
        "kind": QuickResponseType.THANKS_EN.value,
        "route_content": "用户在用英文表达感谢，我会直接回应。",
        "content": "You’re welcome. Feel free to ask another tendering or procurement-related question.",
    },
    QuickResponseType.GOODBYE_ZH: {
        "kind": QuickResponseType.GOODBYE_ZH.value,
        "route_content": "用户在结束对话，我会直接回应。",
        "content": "再见！如有招投标相关问题，欢迎随时回来咨询。",
    },
    QuickResponseType.GOODBYE_EN: {
        "kind": QuickResponseType.GOODBYE_EN.value,
        "route_content": "用户在用英文结束对话，我会直接回应。",
        "content": "Goodbye! You’re welcome to return whenever you have another tendering or procurement question.",
    },
    QuickResponseType.CAPABILITIES_ZH: {
        "kind": QuickResponseType.CAPABILITIES_ZH.value,
        "route_content": "用户在询问助手能力范围，我会直接说明可处理的问题类型。",
        "content": "我是招投标智能问答助手，可以协助您了解政策法规，并查询和分析招标项目、舆情动态、企业信息和商品资料。您可以直接描述查询对象、地区、时间范围和关注的问题。",
    },
    QuickResponseType.CAPABILITIES_EN: {
        "kind": QuickResponseType.CAPABILITIES_EN.value,
        "route_content": "用户在用英文询问助手能力范围，我会直接说明可处理的问题类型。",
        "content": "I’m a tendering and procurement Q&A assistant. I can help with policies, tenders, public opinion, companies, and products. You can specify the subject, region, time range, and information you need.",
    },
    QuickResponseType.WELLBEING_ZH: {
        "kind": QuickResponseType.WELLBEING_ZH.value,
        "route_content": "用户在询问运行状态，我会直接回应。",
        "content": "谢谢关心，我运行正常，可以随时为您处理招投标相关问题。",
    },
    QuickResponseType.WELLBEING_EN: {
        "kind": QuickResponseType.WELLBEING_EN.value,
        "route_content": "用户在用英文询问运行状态，我会直接回应。",
        "content": "Thank you for asking. I’m operating normally and ready to help with tendering and procurement questions.",
    },
    QuickResponseType.ACKNOWLEDGEMENT_ZH: {
        "kind": QuickResponseType.ACKNOWLEDGEMENT_ZH.value,
        "route_content": "用户在做简短确认，我会直接回应。",
        "content": "好的。需要继续查询时，请直接告诉我具体问题。",
    },
    QuickResponseType.ACKNOWLEDGEMENT_EN: {
        "kind": QuickResponseType.ACKNOWLEDGEMENT_EN.value,
        "route_content": "用户在用英文做简短确认，我会直接回应。",
        "content": "Understood. When you are ready, tell me what you would like to investigate.",
    },
    QuickResponseType.COMPLIMENT_ZH: {
        "kind": QuickResponseType.COMPLIMENT_ZH.value,
        "route_content": "用户在表达认可，我会直接回应。",
        "content": "谢谢您的认可。我会继续尽力提供准确、清晰的回答。",
    },
    QuickResponseType.COMPLIMENT_EN: {
        "kind": QuickResponseType.COMPLIMENT_EN.value,
        "route_content": "用户在用英文表达认可，我会直接回应。",
        "content": "Thank you. I’ll continue working to provide accurate and clear answers.",
    },
    QuickResponseType.APOLOGY_ZH: {
        "kind": QuickResponseType.APOLOGY_ZH.value,
        "route_content": "用户在表达歉意，我会直接回应。",
        "content": "没关系。您可以继续说明问题或补充需要查询的信息。",
    },
    QuickResponseType.APOLOGY_EN: {
        "kind": QuickResponseType.APOLOGY_EN.value,
        "route_content": "用户在用英文表达歉意，我会直接回应。",
        "content": "No problem. You can continue explaining the question or provide additional details.",
    },
}