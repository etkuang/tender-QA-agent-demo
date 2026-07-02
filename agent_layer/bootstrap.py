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
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.models import ModelFactory
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.schemas import Category
from agent_layer.workflows.child_task import ChildAnswerChain, ChildIntentClassifier, GeneralChildTaskWorkflow
from agent_layer.workflows.common import ChildTaskWorkflowProfile
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
    policy_profile = ChildTaskWorkflowProfile(
        description=(
            "Policy and regulation questions. Prefer local PDF policy evidence retrieved by the RAG system. "
            "Use website evidence only when local policy evidence is missing, stale, or insufficient."
        ),
        tools_pool=["rag", "website"],
        tool_preference=[["rag"], ["website"]],
    )
    policy_assessment = PolicyAssessmentChain(structured_model, runtime_settings)
    policy_query_parser = PolicyQueryParser(structured_model, runtime_settings)
    policy_self_rag = PolicySelfRAGPlugin()
    policy_workflow = PolicyWorkflow(
        policy_profile,
        retrieval_pipeline,
        policy_query_parser,
        policy_assessment,
        answer_model,
        evidence_adapter,
        runtime_settings,
        checkpoint_runtime,
        policy_self_rag,
        policy_internet_client,
    )

    clients = _normalize_website_clients(
        website_clients,
        policy_internet_client,
        general_internet_client,
    )
    data_context_catalog = build_default_data_context_catalog()
    data_context_retriever = DataContextRetriever(data_context_catalog, runtime_settings)
    sql_generator = DataSQLGenerator(structured_model, runtime_settings)
    sql_repair_chain = DataSQLRepairChain(structured_model, runtime_settings)
    sql_validator = SQLPolicyValidator(runtime_settings.sql_dialect, runtime_settings.sql_max_rows)
    web_supplementer = WebsiteSupplementer(evidence_adapter, clients, runtime_settings)
    analyzer = DataResultAnalyzer()
    intent_classifier = ChildIntentClassifier(structured_model, runtime_settings)
    answer_chain = ChildAnswerChain(answer_model)

    profiles = {
        Category.TENDER: ChildTaskWorkflowProfile(
            description=(
                "Tender announcements, project conditions, bid results, regional projects, "
                "and historical tender statistics. Prefer structured SQL records. "
                "Use website evidence only when SQL has no eligible evidence or current public announcements are required."
            ),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.PUBLIC_OPINION: ChildTaskWorkflowProfile(
            description=(
                "Public-opinion, news, regulatory updates, negative events, and risk trend questions. "
                "SQL event clusters and website evidence are both primary evidence sources."
            ),
            tools_pool=["sql", "website"],
            tool_preference=[["sql", "website"]],
        ),
        Category.COMPANY: ChildTaskWorkflowProfile(
            description=(
                "Company profile, qualification, historical bid performance, competitiveness, "
                "and entity risk questions. Prefer structured SQL records. "
                "Use website evidence only when SQL has no eligible evidence or current public status is required."
            ),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.PRODUCT: ChildTaskWorkflowProfile(
            description=(
                "Product brand, model, parameters, comparable-product analysis, and procurement fit questions. "
                "Prefer structured product catalog SQL records. "
                "Use website evidence only when SQL has no eligible evidence or manufacturer/public page freshness is required."
            ),
            tools_pool=["sql", "website"],
            tool_preference=[["sql"], ["website"]],
        ),
        Category.OTHER: ChildTaskWorkflowProfile(
            description=(
                "General questions outside the supported tender, policy, public-opinion, company, and product domains. "
                "Use website evidence first when available. Fall back to model-only answering without claiming tool-backed evidence."
            ),
            tools_pool=["website", "model_only"],
            tool_preference=[["website"], ["model_only"]],
        ),
    }
    child_task_workflows = {
        category: GeneralChildTaskWorkflow(
            category,
            profile,
            intent_classifier,
            sql_generator,
            sql_repair_chain,
            sql_validator,
            sql_gateway,
            data_context_retriever,
            web_supplementer,
            analyzer,
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
