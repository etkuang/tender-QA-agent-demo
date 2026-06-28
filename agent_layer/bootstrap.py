# coding: utf-8

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.question_decomposition.chain import QuestionDecomposer
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.config import Settings, settings
from agent_layer.domains.company import build_company_profile
from agent_layer.domains.price import build_price_profile
from agent_layer.domains.product import build_product_profile
from agent_layer.domains.public_opinion import build_public_opinion_profile
from agent_layer.domains.tender import build_tender_profile
from agent_layer.models import ModelFactory
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.sql.gateway import SQLGateway
from agent_layer.workflows.analysis import DatasetAnalyzer
from agent_layer.workflows.data_domain import DataDomainWorkflow, ResearchPlanner
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

    planner = ResearchPlanner(structured_model, runtime_settings)
    clients = website_clients or {}
    profiles = [
        build_tender_profile(runtime_settings),
        build_public_opinion_profile(runtime_settings),
        build_company_profile(runtime_settings),
        build_price_profile(runtime_settings),
        build_product_profile(runtime_settings),
    ]
    profile_map = {profile.category: profile for profile in profiles}
    analyzer = DatasetAnalyzer()
    data_workflows = {
        profile.category: DataDomainWorkflow(
            profile,
            profile_map,
            planner,
            answer_model,
            evidence_adapter,
            runtime_settings,
            sql_gateway,
            clients,
            analyzer,
        )
        for profile in profiles
    }

    dependencies = ApplicationDependencies(
        settings=runtime_settings,
        quick_classifier=quick_classifier,
        decomposer=decomposer,
        general_workflow=general_workflow,
        composite_workflow=composite_workflow,
        policy_workflow=policy_workflow,
        data_workflows=data_workflows,
        checkpoint_runtime=checkpoint_runtime,
    )
    return TenderQAApplication(dependencies)
