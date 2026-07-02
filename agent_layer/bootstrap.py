# coding: utf-8

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.question_decomposition.chain import QuestionDecomposer
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.config import Settings, settings
from agent_layer.data_domain.analysis import DataResultAnalyzer
from agent_layer.data_domain.chains import (
    DataAnswerChain,
    DataIntentClassifier,
    DataSQLGenerator,
    DataSQLRepairChain,
)
from agent_layer.data_domain.context.catalog import build_default_data_context_catalog
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.data_domain.workflow import DataDomainWorkflow
from agent_layer.domains.company import build_company_profile
from agent_layer.domains.price import build_price_profile
from agent_layer.domains.product import build_product_profile
from agent_layer.domains.public_opinion import build_public_opinion_profile
from agent_layer.domains.tender import build_tender_profile
from agent_layer.models import ModelFactory
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.schemas import Category
from agent_layer.workflows.general import CompositeAnswerWorkflow, GeneralWorkflow
from agent_layer.workflows.policy import PolicyAssessmentChain, PolicyQueryParser
from agent_layer.workflows.policy_graph import PolicyGraphWorkflow


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
    general_workflow = GeneralWorkflow(
        answer_model,
        runtime_settings,
        evidence_adapter,
        general_internet_client,
    )
    composite_workflow = CompositeAnswerWorkflow(answer_model, runtime_settings)
    policy_assessment = PolicyAssessmentChain(structured_model, runtime_settings)
    policy_query_parser = PolicyQueryParser(structured_model, runtime_settings)
    policy_workflow = PolicyGraphWorkflow(
        retrieval_pipeline,
        policy_query_parser,
        policy_assessment,
        answer_model,
        evidence_adapter,
        runtime_settings,
        checkpoint_runtime,
        policy_internet_client,
    )

    clients = website_clients or {}
    profiles = [
        build_tender_profile(runtime_settings),
        build_public_opinion_profile(runtime_settings),
        build_company_profile(runtime_settings),
        build_price_profile(runtime_settings),
        build_product_profile(runtime_settings),
    ]
    profile_map = {profile.category: profile for profile in profiles}
    data_context_catalog = build_default_data_context_catalog()
    data_context_retriever = DataContextRetriever(data_context_catalog, runtime_settings)
    data_intent_classifier = DataIntentClassifier(structured_model, runtime_settings)
    data_sql_generator = DataSQLGenerator(structured_model, runtime_settings)
    data_sql_repair_chain = DataSQLRepairChain(structured_model, runtime_settings)
    data_sql_validator = SQLPolicyValidator(runtime_settings.sql_dialect, runtime_settings.sql_max_rows)
    data_answer_chain = DataAnswerChain(answer_model, runtime_settings)
    data_web_supplementer = WebsiteSupplementer(evidence_adapter, clients, runtime_settings)
    data_analyzer = DataResultAnalyzer()
    data_workflows = {
        profile.category: DataDomainWorkflow(
            profile,
            profile_map,
            data_context_retriever,
            data_intent_classifier,
            data_sql_generator,
            data_sql_repair_chain,
            data_sql_validator,
            sql_gateway,
            data_answer_chain,
            data_web_supplementer,
            data_analyzer,
            runtime_settings,
        )
        for profile in profiles
    }
    child_task_workflows = {
        Category.OTHER: general_workflow,
        Category.POLICY: policy_workflow,
        **data_workflows,
    }

    dependencies = ApplicationDependencies(
        settings=runtime_settings,
        quick_classifier=quick_classifier,
        decomposer=decomposer,
        child_task_workflows=child_task_workflows,
        composite_workflow=composite_workflow,
        checkpoint_runtime=checkpoint_runtime,
    )
    return TenderQAApplication(dependencies)
