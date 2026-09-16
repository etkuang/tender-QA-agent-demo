# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from agent_layer.config import settings

QUICK_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是一个招投标 AI 助手的快捷回复类别分类器，判断当前用户消息是否只属于可以固定话术回复的类别。

请按以下 JSON 格式返回：
{{"quick_response_type": "none"}}

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
- 无法判断时选择 none。"""
    ),

    HumanMessagePromptTemplate.from_template(
        "用户最新消息：\n{last_user_message}"
    ),
])

QUESTION_DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是一个招投标 AI 助手的问题分类与拆解器。根据历史消息语境，将用户最新问题拆成一个或多个child task。

请按以下 JSON 格式返回：
{{
  "child_tasks": [
    {{
      "task_id": "q1",
      "question": "子问题内容",
      "category": "policy",
      "depends_on": [],
      "requires_fresh_data": false,
      "clarification_question": null
    }}
  ]
}}

child task 是一个可执行的子问题，用于决定后续调用哪个工作流处理。
字段说明：
- child_tasks：包含一个或多个 child task 对象的数组。
- task_id：使用 q1、q2、q3 这样的稳定短 ID。
- question：当前子问题本身。若用户问题包含指代词或省略表达，将指代词替换成历史消息中的具体名词实体，并补充被省略的实体。
- category：当前子问题所属类别，只能选择一个类别。
- depends_on：当前子问题依赖的前置 child task ID；没有依赖时使用空列表。
- requires_fresh_data：只有用户询问近期、最新、当前公告、现状、实时报价或舆情时才为 true。
- clarification_question：仅当 category 为 unclear 时填写需要追问用户的问题；其他类别使用 null。

category 只能取以下值之一：
- policy：法律法规、政策依据、违法性、处罚、程序规则、条款解释、效力与适用范围。
- tender：招标公告、项目条件、中标结果、地区项目、历史项目统计。
- public_opinion：项目或企业的新闻、监管动态、负面事件、舆情趋势。
- company：企业画像、资质、历史中标能力、竞争力、主体风险。
- product：品牌型号、商品参数、同类比较、采购适配性。
- other：用户意图清楚，但与以上业务域无关，可由通用问答处理。
- unclear：用户意图、业务对象或业务域不足以判断，需要补充信息。

拆解规则：
- 如果用户最新问题包含多个业务目标，拆成多个 child task；同一类 child task 可以有多个，取决于包含多少个主体。
- 如果后续 child task 需要前序 child task 的实体、结论或证据，在 depends_on 中填写前序 task_id。
- depends_on 只能引用已经生成的前序 task_id。
- 历史助手回答只用于理解对话语境，不作为事实依据补造用户没有确认的信息。"""
    ),

    HumanMessagePromptTemplate.from_template(
        "历史消息：\n{history}\n\n用户最新问题：\n{question}"
    ),
])

OTHER_CONTEXT_TEMPLATE = """## 问题

{question}

## 时效要求

{freshness_requirement}

## 对话背景

{history}

## 前置问题及答案

{prerequisite_answers}

## 可用证据

{evidence}"""


OTHER_RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是通用问题资料研究助手。

研究要求：
- 根据所给问题、时效要求、对话背景、前置问题及答案和可用证据，判断是否还需要检索互联网资料。
- 对话背景只用于理解指代、上下文和用户已经明确的条件；前置问题及答案用于承接已有结论，其中涉及的外部事实只有在“可用证据”中存在对应证据时才视为已有资料。
- 参考“可用证据”中的可靠性说明判断现有资料是否足够。
- 当问题需要最新信息而可用证据不足，或现有资料不足以可靠回答问题时，调用 internet_search。
- 当现有资料已经足够，或问题不需要外部资料即可使用稳定通用知识回答时，不调用工具。
- 每轮最多调用一次 internet_search；检索查询应聚焦问题中的具体主体、主题、地域、时间、型号或政策名称。
- 只负责资料研究和工具选择，不生成给用户的最终回答。"""
    ),
    HumanMessagePromptTemplate.from_template(OTHER_CONTEXT_TEMPLATE),
])


OTHER_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是通用问题回答助手。

回答要求：
- 直接、完整地回答所给问题。
- 可用信息范围：对话背景只用于理解指代、上下文和用户已经明确的条件；前置问题及答案用于承接已有结论，其中涉及的外部事实只有在“可用证据”中存在对应证据时才可使用；需要最新信息时只使用“可用证据”，不需要最新信息时可以使用稳定的通用知识和“可用证据”；不要使用这些范围之外的信息。
- 在结构化输出的 evidence_ids 中只列出本次回答实际使用的 evidence_id，每个 evidence_id 最多出现一次。
- 如果可用证据不足或相互冲突，请明确说明；不能形成可靠答案时，将 status 设为 insufficient_evidence 并说明原因。

请按以下 JSON 格式返回：
{{"status": "answered", "answer": "完整回答", "evidence_ids": []}}

status 只能是 answered 或 insufficient_evidence；answer 是完整回答或证据不足说明；evidence_ids 是实际使用的证据标识列表。"""
    ),
    HumanMessagePromptTemplate.from_template(OTHER_CONTEXT_TEMPLATE),
])


WEBSITE_EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是网页证据整理助手。

整理要求：
- 根据问题，从提供的网页资料中提取可用于回答问题的证据，返回 evidence 列表；没有可用证据时返回空列表。
- 最多返回 {evidence_limit} 条与问题相关的证据。
- 每条证据的 title 不超过 {title_max_chars} 个字符，content 不超过 {content_max_chars} 个字符，reliability_description 不超过 {reliability_max_chars} 个字符；字符数包含标点、空格和换行。
- 每条证据聚焦一个主题；资料过长或包含多个主题时，生成多条符合长度限制的独立证据，将紧密相关的事实及其条件、例外保留在同一条证据中，不必将每个事实拆成单独一条。
- title 简洁概括该条证据的主题；content 必须能够脱离 title、所给问题、其他证据和原始对话被独立理解。
- content 明确写出相关主体、项目、政策或产品；保留理解事实所必需且网页资料明确支持的时间、地域、适用范围、单位、条件、例外和陈述来源。
- 仅依据所给网页资料补全指代和省略信息；问题只用于筛选相关资料，不作为补充事实的依据。资料没有说明的背景不自行补充；保留报道、声明或指控等原有表述性质，不将其改写为已确认事实。
- source_type 固定为 website；url 使用对应资料的网页链接；updated_at 使用资料明确给出的更新时间，没有时填写 null。
- reliability_description 说明该条资料的可靠性依据与局限；无法从所给资料判断的部分应明确说明。
- 只返回结构化输出要求的字段，不生成关键词、证据标识或向量。"""
    ),
    HumanMessagePromptTemplate.from_template(
        "## 问题\n\n{question}\n\n## 网页资料\n\n{web_materials}"
    ),
]).partial(
    title_max_chars=settings.evidence_title_max_chars,
    content_max_chars=settings.evidence_content_max_chars,
    reliability_max_chars=settings.evidence_reliability_max_chars,
)