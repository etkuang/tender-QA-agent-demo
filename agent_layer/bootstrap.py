# coding: utf-8

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.question_decomposition.chain import QuestionDecomposer
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.config import Settings, settings
from agent_layer.data_domain.analysis import DataResultAnalyzer
from agent_layer.data_domain.chains import DataSQLGenerator, DataSQLRepairChain
from agent_layer.data_domain.context.catalog import build_default_data_context_catalog
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.workflows.tools import (
    ChildTaskToolRegistry,
    ModelOnlyChildTaskTool,
    RAGChildTaskTool,
    SQLChildTaskTool,
    WebsiteChildTaskTool,
)
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.models import ModelFactory
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.schemas import Category
from agent_layer.workflows.child_task import ChildAnswerChain, ChildIntentClassifier, GeneralChildTaskWorkflow
from agent_layer.workflows.common import ChildTaskQueryRoute, ChildTaskWorkflowProfile
from agent_layer.workflows.general import CompositeAnswerWorkflow
from agent_layer.workflows.policy import PolicyAssessmentChain, PolicyQueryParser
from agent_layer.workflows.policy_graph import PolicyWorkflow
from agent_layer.workflows.self_rag import PolicySelfRAGPlugin


async def build_application(
    runtime_settings: Settings = settings,
    sql_gateway: SQLGateway | None = None,
    website_clients: dict[str, WebsiteSearchClient] | None = None,
    policy_internet_client: WebsiteSearchClient | None = None,
    general_internet_client: WebsiteSearchClient | None = None,
) -> TenderQAApplication:
    model_factory = ModelFactory(runtime_settings)
    structured_model = model_factory.build_chat_model(temperature=0.0, max_tokens=1000)
    answer_model = model_factory.build_chat_model(temperature=0.2)
    checkpoint_runtime = await LangGraphCheckpointRuntime.open(runtime_settings.checkpoint_path)

    evidence_adapter = EvidenceAdapter()
    retrieval_pipeline = RetrievalPipeline(
        KnowledgeBaseClient(runtime_settings),
        evidence_adapter,
        runtime_settings,
    )

    decomposer = QuestionDecomposer(structured_model, runtime_settings)
    quick_classifier = QuickResponseClassifier(structured_model, runtime_settings)
    composite_workflow = CompositeAnswerWorkflow(answer_model, runtime_settings)
    policy_assessment = PolicyAssessmentChain(structured_model, runtime_settings)
    policy_query_parser = PolicyQueryParser(structured_model, runtime_settings)
    policy_self_rag = PolicySelfRAGPlugin()

    clients = _normalize_website_clients(
        website_clients,
        policy_internet_client,
        general_internet_client,
    )
    data_context_catalog = build_default_data_context_catalog()
    data_context_retriever = DataContextRetriever(data_context_catalog, runtime_settings)
    sql_generator = DataSQLGenerator(structured_model, runtime_settings)
    sql_repair_chain = DataSQLRepairChain(structured_model, runtime_settings)
    sql_validator = SQLPolicyValidator()
    web_supplementer = WebsiteSupplementer(evidence_adapter, clients, runtime_settings)
    analyzer = DataResultAnalyzer()
    answer_chain = ChildAnswerChain(answer_model)
    tool_registry = ChildTaskToolRegistry(
        {
            "rag": RAGChildTaskTool(retrieval_pipeline),
            "sql": SQLChildTaskTool(
                sql_generator,
                sql_repair_chain,
                sql_validator,
                sql_gateway,
                data_context_retriever,
                analyzer,
                runtime_settings,
            ),
            "website": WebsiteChildTaskTool(web_supplementer),
            "model_only": ModelOnlyChildTaskTool(),
        }
    )
    policy_workflow = PolicyWorkflow(
        tool_registry.subset(["rag", "website"]),
        policy_query_parser,
        policy_assessment,
        answer_model,
        runtime_settings,
        checkpoint_runtime,
        policy_self_rag,
    )

    profiles = {
        Category.TENDER: ChildTaskWorkflowProfile(
            description="招标公告、项目条件、中标结果、地区项目和历史招投标统计。",
            query_routes=_data_query_routes(Category.TENDER),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.PUBLIC_OPINION: ChildTaskWorkflowProfile(
            description="项目或企业的新闻、监管动态、负面事件和舆情趋势。",
            query_routes=_data_query_routes(Category.PUBLIC_OPINION),
            tools_pool=["sql", "website"],
            tool_preference=[["sql", "website"]],
        ),
        Category.COMPANY: ChildTaskWorkflowProfile(
            description="企业画像、资质、历史投标表现、竞争力和主体风险。",
            query_routes=_data_query_routes(Category.COMPANY),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.PRODUCT: ChildTaskWorkflowProfile(
            description="产品品牌、型号、参数、同类比较和采购适配分析。",
            query_routes=_data_query_routes(Category.PRODUCT),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.OTHER: ChildTaskWorkflowProfile(
            description="政策、招投标、舆情、企业和产品领域以外的明确通用问题。",
            query_routes=_other_query_routes(),
            tools_pool=["website", "model_only"],
            tool_preference=[["website"], ["model_only"]],
        ),
    }
    child_task_workflows = {
        category: GeneralChildTaskWorkflow(
            category,
            profile,
            ChildIntentClassifier(
                structured_model,
                runtime_settings,
                category,
                profile,
            ),
            tool_registry.subset(profile.tools_pool),
            answer_chain,
            runtime_settings,
        )
        for category, profile in profiles.items()
    }
    child_task_workflows[Category.POLICY] = policy_workflow

    dependencies = ApplicationDependencies(
        settings=runtime_settings,
        quick_classifier=quick_classifier,
        decomposer=decomposer,
        child_task_workflows=child_task_workflows,
        composite_workflow=composite_workflow,
        checkpoint_runtime=checkpoint_runtime,
    )
    return TenderQAApplication(dependencies)


def _data_query_routes(category: Category) -> list[ChildTaskQueryRoute]:
    routes = {
        Category.TENDER: [
            ChildTaskQueryRoute(
                route_name="project_lookup",
                definition="查询招标公告、采购项目、项目条件、地区项目或公告状态。",
                query_prompt="生成独立的项目查询句，明确项目或关键词、地区、公告时间、状态、筛选条件、所需字段和返回数量；不得补造未指定值。",
            ),
            ChildTaskQueryRoute(
                route_name="bid_result_lookup",
                definition="查询中标结果、中标人、中标金额或项目成交信息。",
                query_prompt="生成独立的中标结果查询句，明确项目、采购人或中标人、时间范围、金额口径、所需字段和排序；不得把预算当作中标金额。",
            ),
            ChildTaskQueryRoute(
                route_name="tender_statistics",
                definition="统计、排名、比较或分析招投标项目与中标结果的时间变化。",
                query_prompt="生成完整的招投标统计查询句，明确统计指标及定义、聚合函数、分组维度、时间字段与范围、排序、top-N 和空值处理。",
            ),
        ],
        Category.PUBLIC_OPINION: [
            ChildTaskQueryRoute(
                route_name="event_lookup",
                definition="查询企业或项目的新闻、监管动态、负面事件或具体舆情记录。",
                query_prompt="生成独立的舆情事件查询句，明确主体、事件类型、时间窗口、来源范围、情绪方向、去重口径和返回字段。",
            ),
            ChildTaskQueryRoute(
                route_name="trend_analysis",
                definition="统计或比较舆情事件数量、情绪方向、来源或时间趋势。",
                query_prompt="生成完整的舆情趋势查询句，明确主体、时间窗口与粒度、事件去重口径、分组维度、比较基准和排序。",
            ),
            ChildTaskQueryRoute(
                route_name="risk_assessment",
                definition="根据新闻、监管和负面事件分析主体风险。",
                query_prompt="生成风险证据查询句，明确主体、风险类别、时间窗口和所需事件证据，并要求区分可验证事实、趋势关联和不能确认的因果判断。",
            ),
        ],
        Category.COMPANY: [
            ChildTaskQueryRoute(
                route_name="company_profile",
                definition="查询企业基础画像、主体信息或业务覆盖范围。",
                query_prompt="生成独立的企业画像查询句，明确可消歧的企业标识、所需字段、数据时点和返回范围；名称不能稳定消歧时不得自行合并。",
            ),
            ChildTaskQueryRoute(
                route_name="qualification_check",
                definition="查询企业资质、资格条件或公开状态。",
                query_prompt="生成企业资质核验查询句，明确企业、资质名称、有效时间、地区或主管范围以及需要数据库还是当前公开来源核验。",
            ),
            ChildTaskQueryRoute(
                route_name="bid_performance",
                definition="查询或统计企业历史投标与中标表现。",
                query_prompt="生成企业投标表现查询句，明确企业、时间范围、地区或品类、项目数量、金额口径、分组方式和排序要求。",
            ),
            ChildTaskQueryRoute(
                route_name="competitiveness_analysis",
                definition="比较企业竞争力、市场覆盖或历史中标能力。",
                query_prompt="生成企业竞争力比较查询句，明确比较企业、统一指标定义、样本范围、时间窗口、分组和排名规则，不得把相关性写成因果。",
            ),
            ChildTaskQueryRoute(
                route_name="company_risk",
                definition="查询企业经营异常、处罚、舆情或其他主体风险。",
                query_prompt="生成企业风险证据查询句，明确企业、风险类别、时间窗口、所需数据库字段和当前公开核验要求。",
            ),
        ],
        Category.PRODUCT: [
            ChildTaskQueryRoute(
                route_name="product_lookup",
                definition="查询产品品牌、型号、参数、来源或更新时间。",
                query_prompt="生成独立的产品查询句，明确品牌与型号、目标参数、参数单位、版本或更新时间和返回字段；不得猜测型号。",
            ),
            ChildTaskQueryRoute(
                route_name="product_comparison",
                definition="比较多个产品或型号的参数、价格或适用差异。",
                query_prompt="生成产品比较查询句，明确待比较型号、统一参数与单位、价格时点和地区、缺失值处理及输出顺序。",
            ),
            ChildTaskQueryRoute(
                route_name="procurement_fit",
                definition="判断产品是否满足采购需求或推荐可比较产品。",
                query_prompt="生成采购适配查询句，明确必须满足的参数、可选条件、排除条件、采购场景和候选范围，并要求逐项说明证据与缺失参数。",
            ),
        ],
    }
    terminal_routes = [
        ChildTaskQueryRoute(
            route_name="clarification",
            definition="业务术语、实体、时间、指标或比较口径存在会改变查询结果的关键歧义。",
            query_prompt="生成一个简短、具体且只询问必要缺失信息的用户澄清问题。",
        ),
        ChildTaskQueryRoute(
            route_name="unsupported",
            definition="涉及写操作、权限绕过、敏感信息提取，或明显超出当前工作流能力。",
            query_prompt="生成一个简明的中文说明，指出具体的安全限制或不支持原因，不提供规避方法。",
        ),
    ]
    return [*routes[category], *terminal_routes]


def _other_query_routes() -> list[ChildTaskQueryRoute]:
    return [
        ChildTaskQueryRoute(
            route_name="general_answer",
            definition="不依赖实时信息即可回答的明确通用知识问题。",
            query_prompt="把任务改写为一个完整、具体的回答指令，明确回答范围、所需解释深度和输出形式。",
        ),
        ChildTaskQueryRoute(
            route_name="current_information",
            definition="需要近期、最新、当前状态或公开网页核验的通用问题。",
            query_prompt="把任务改写为一个需要公开来源核验的检索指令，明确主题、地区、时间窗口和信息时点。",
        ),
        ChildTaskQueryRoute(
            route_name="follow_up",
            definition="依赖最近对话或前置子任务语境的通用追问。",
            query_prompt="补全历史语境中已经明确的对象和限制条件，生成能够独立回答的完整指令。",
        ),
        ChildTaskQueryRoute(
            route_name="clarification",
            definition="缺少会实质改变答案的对象、范围或约束。",
            query_prompt="生成一个简短、具体且只询问必要缺失信息的用户澄清问题。",
        ),
        ChildTaskQueryRoute(
            route_name="unsupported",
            definition="涉及危险操作、权限绕过、敏感信息提取或系统明确不支持的请求。",
            query_prompt="生成一个简明的中文说明，指出具体的安全限制或不支持原因，不提供规避方法。",
        ),
    ]


def _normalize_website_clients(
    website_clients: dict[str, WebsiteSearchClient] | None,
    policy_internet_client: WebsiteSearchClient | None,
    general_internet_client: WebsiteSearchClient | None,
) -> dict[str, WebsiteSearchClient]:
    clients = dict(website_clients or {})
    if policy_internet_client is not None:
        clients[Category.POLICY.value] = policy_internet_client
    if general_internet_client is not None:
        clients[Category.OTHER.value] = general_internet_client
    legacy_names = {
        "tender_web": Category.TENDER.value,
        "public_opinion_web": Category.PUBLIC_OPINION.value,
        "company_web": Category.COMPANY.value,
        "product_web": Category.PRODUCT.value,
    }
    for old_name, category_name in legacy_names.items():
        if old_name in clients and category_name not in clients:
            clients[category_name] = clients[old_name]
    return clients
