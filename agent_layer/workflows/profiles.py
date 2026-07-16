# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.schemas import Category
from agent_layer.workflows.common import ChildTaskQueryType, ChildTaskWorkflowProfile


def build_general_child_task_profiles() -> list[ChildTaskWorkflowProfile]:
    return [
        ChildTaskWorkflowProfile(
            category=Category.TENDER,
            query_types=_data_query_types(Category.TENDER),
            tool_preference=[["sql"], ["website"]],
        ),
        ChildTaskWorkflowProfile(
            category=Category.PUBLIC_OPINION,
            query_types=_data_query_types(Category.PUBLIC_OPINION),
            tool_preference=[["sql", "website"]],
        ),
        ChildTaskWorkflowProfile(
            category=Category.COMPANY,
            query_types=_data_query_types(Category.COMPANY),
            tool_preference=[["sql"], ["website"]],
        ),
        ChildTaskWorkflowProfile(
            category=Category.PRODUCT,
            query_types=_data_query_types(Category.PRODUCT),
            tool_preference=[["sql"], ["website"]],
        ),
        ChildTaskWorkflowProfile(
            category=Category.OTHER,
            query_types=_other_query_types(),
            tool_preference=[["website"]],
            model_only_fallback=True,
        ),
    ]


def _data_query_types(category: Category) -> list[ChildTaskQueryType]:
    query_types = {
        Category.TENDER: [
            ChildTaskQueryType(
                type_name="project_lookup",
                description="查询招标公告、采购项目、项目条件、地区项目或公告状态。",
                query_format="查询目标：招标或采购项目；项目名称或关键词：<项目名称或关键词>；地区：<地区>；公告时间：<公告时间范围>；项目状态：<状态>；筛选条件：<其他筛选条件>；返回字段：<所需字段>；排序：<排序要求>；返回数量：<返回数量>。",
            ),
            ChildTaskQueryType(
                type_name="bid_result_lookup",
                description="查询中标结果、中标人、中标金额或项目成交信息。",
                query_format="查询目标：中标或成交结果；项目：<项目名称或标识>；采购人：<采购人>；中标人：<中标人>；时间范围：<时间范围>；金额口径：<中标或成交金额口径>；筛选条件：<其他筛选条件>；返回字段：<所需字段>；排序：<排序要求>；返回数量：<返回数量>。预算不得作为中标金额。",
            ),
            ChildTaskQueryType(
                type_name="tender_statistics",
                description="统计、排名、比较或分析招投标项目与中标结果的时间变化。",
                query_format="查询目标：招投标统计；统计指标：<统计指标>；指标定义：<指标定义>；聚合方式：<聚合方式>；分组维度：<分组维度>；时间字段：<时间字段>；时间范围：<时间范围>；筛选条件：<筛选条件>；排序：<排序要求>；Top-N：<数量>；空值处理：<空值处理规则>。",
            ),
        ],
        Category.PUBLIC_OPINION: [
            ChildTaskQueryType(
                type_name="event_lookup",
                description="查询企业或项目的新闻、监管动态、负面事件或具体舆情记录。",
                query_format="查询目标：舆情事件；主体：<企业或项目>；事件类型：<事件类型>；时间窗口：<时间窗口>；来源范围：<来源范围>；情绪方向：<情绪方向>；去重口径：<去重口径>；返回字段：<所需字段>；排序：<排序要求>；返回数量：<返回数量>。",
            ),
            ChildTaskQueryType(
                type_name="trend_analysis",
                description="统计或比较舆情事件数量、情绪方向、来源或时间趋势。",
                query_format="查询目标：舆情趋势；主体：<企业或项目>；时间窗口：<时间窗口>；时间粒度：<时间粒度>；统计指标：<统计指标>；事件去重口径：<去重口径>；分组维度：<分组维度>；比较基准：<比较基准>；排序：<排序要求>。",
            ),
            ChildTaskQueryType(
                type_name="risk_assessment",
                description="根据新闻、监管和负面事件分析主体风险。",
                query_format="查询目标：舆情风险证据；主体：<企业或项目>；风险类别：<风险类别>；时间窗口：<时间窗口>；所需事件证据：<事件证据类型>；来源范围：<来源范围>；输出要求：区分可验证事实、趋势关联和不能确认的因果判断。",
            ),
        ],
        Category.COMPANY: [
            ChildTaskQueryType(
                type_name="company_profile",
                description="查询企业基础画像、主体信息或业务覆盖范围。",
                query_format="查询目标：企业画像；企业名称：<企业名称>；统一社会信用代码或其他标识：<企业标识>；返回字段：<所需字段>；数据时点：<数据时点>；返回范围：<返回范围>。企业名称不能稳定消歧时不得自行合并。",
            ),
            ChildTaskQueryType(
                type_name="qualification_check",
                description="查询企业资质、资格条件或公开状态。",
                query_format="查询目标：企业资质核验；企业：<企业名称或标识>；资质名称：<资质名称>；有效时间：<有效时间>；地区或主管范围：<地区或主管范围>；核验时点：<核验时点>；来源要求：<数据库或当前公开来源>；返回字段：<所需字段>。",
            ),
            ChildTaskQueryType(
                type_name="bid_performance",
                description="查询或统计企业历史投标与中标表现。",
                query_format="查询目标：企业投标与中标表现；企业：<企业名称或标识>；时间范围：<时间范围>；地区：<地区>；品类：<品类>；统计指标：<项目数量或金额等指标>；金额口径：<金额口径>；分组方式：<分组方式>；排序：<排序要求>；Top-N：<数量>。",
            ),
            ChildTaskQueryType(
                type_name="competitiveness_analysis",
                description="比较企业竞争力、市场覆盖或历史中标能力。",
                query_format="查询目标：企业竞争力比较；比较企业：<企业列表>；比较指标：<指标列表>；指标定义：<统一指标定义>；样本范围：<样本范围>；时间窗口：<时间窗口>；分组方式：<分组方式>；排名规则：<排名规则>。不得把相关性写成因果。",
            ),
            ChildTaskQueryType(
                type_name="company_risk",
                description="查询企业经营异常、处罚、舆情或其他主体风险。",
                query_format="查询目标：企业风险证据；企业：<企业名称或标识>；风险类别：<风险类别>；时间窗口：<时间窗口>；数据库字段：<所需数据库字段>；当前公开核验：<需要核验的当前事实>；来源要求：<来源要求>；返回字段：<所需字段>。",
            ),
        ],
        Category.PRODUCT: [
            ChildTaskQueryType(
                type_name="product_lookup",
                description="查询产品品牌、型号、参数、来源或更新时间。",
                query_format="查询目标：产品信息；品牌：<品牌>；型号：<型号>；目标参数：<参数列表>；参数单位：<单位要求>；版本或更新时间：<版本或更新时间>；返回字段：<所需字段>；返回数量：<返回数量>。不得猜测型号。",
            ),
            ChildTaskQueryType(
                type_name="product_comparison",
                description="比较多个产品或型号的参数、价格或适用差异。",
                query_format="查询目标：产品比较；比较产品：<品牌和型号列表>；比较参数：<统一参数列表>；参数单位：<统一单位>；价格时点：<价格时点>；价格地区：<价格地区>；缺失值处理：<缺失值处理规则>；输出顺序：<输出顺序>。",
            ),
            ChildTaskQueryType(
                type_name="procurement_fit",
                description="判断产品是否满足采购需求或推荐可比较产品。",
                query_format="查询目标：产品采购适配；必须满足的参数：<必选参数>；可选条件：<可选条件>；排除条件：<排除条件>；采购场景：<采购场景>；候选范围：<品牌或型号范围>；输出要求：逐项说明证据和缺失参数。",
            ),
        ],
    }
    terminal_types = [
        ChildTaskQueryType(
            type_name="clarification",
            description="业务术语、实体、时间、指标或比较口径存在会改变查询结果的关键歧义。",
            query_format="生成一个简短、具体且只询问必要缺失信息的用户澄清问题。",
        ),
        ChildTaskQueryType(
            type_name="unsupported",
            description="涉及写操作、权限绕过、敏感信息提取，或明显超出当前工作流能力。",
            query_format="生成一个简明的中文说明，指出具体的安全限制或不支持原因，不提供规避方法。",
        ),
    ]
    return [*query_types[category], *terminal_types]


def _other_query_types() -> list[ChildTaskQueryType]:
    return [
        ChildTaskQueryType(
            type_name="general_answer",
            description="不依赖实时信息即可回答的明确通用知识问题。",
            query_format="查询目标：通用知识回答；主题：<主题>；回答范围：<回答范围>；解释深度：<解释深度>；输出形式：<输出形式>。",
        ),
        ChildTaskQueryType(
            type_name="current_information",
            description="需要近期、最新、当前状态或公开网页核验的通用问题。",
            query_format="查询目标：当前公开信息核验；主题：<主题>；地区：<地区>；时间窗口：<时间窗口>；信息时点：<信息时点>；来源要求：<来源要求>；输出字段：<所需信息>。",
        ),
        ChildTaskQueryType(
            type_name="follow_up",
            description="依赖最近对话或前置子任务语境的通用追问。",
            query_format="查询目标：语境追问；历史语境中的对象：<对象>；已明确的限制条件：<限制条件>；当前追问：<当前追问>；输出要求：<输出要求>。",
        ),
        ChildTaskQueryType(
            type_name="clarification",
            description="缺少会实质改变答案的对象、范围或约束。",
            query_format="生成一个简短、具体且只询问必要缺失信息的用户澄清问题。",
        ),
        ChildTaskQueryType(
            type_name="unsupported",
            description="涉及危险操作、权限绕过、敏感信息提取或系统明确不支持的请求。",
            query_format="生成一个简明的中文说明，指出具体的安全限制或不支持原因，不提供规避方法。",
        ),
    ]