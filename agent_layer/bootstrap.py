# coding: utf-8

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.classification.chain import QuestionClassifier
from agent_layer.checkpoint import SQLiteCheckpointStore
from agent_layer.config import Settings, settings
from agent_layer.context import ContextResolver
from agent_layer.domains.company import build_company_profile
from agent_layer.domains.price import build_price_profile
from agent_layer.domains.product import build_product_profile
from agent_layer.domains.public_opinion import build_public_opinion_profile
from agent_layer.domains.tender import build_tender_profile
from agent_layer.models import ModelFactory
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.retrieval.rewrite import QuestionRewriter
from agent_layer.sql.gateway import SQLGateway
from agent_layer.workflows.analysis import DatasetAnalyzer
from agent_layer.workflows.data_domain import DataDomainWorkflow, ResearchPlanner
from agent_layer.workflows.general import GeneralWorkflow
from agent_layer.workflows.policy import PolicyAssessmentChain, PolicyQueryParser, PolicyWorkflow
from agent_layer.workflows.router import WorkflowRouter


def build_application(
    runtime_settings: Settings = settings,
    sql_gateway: SQLGateway | None = None,
    website_clients: dict[str, WebsiteSearchClient] | None = None,
    policy_internet_client: WebsiteSearchClient | None = None,
) -> TenderQAApplication:
    model_factory = ModelFactory(runtime_settings)
    structured_model = model_factory.build_chat_model(temperature=0.0, max_tokens=1000)
    answer_model = model_factory.build_chat_model(temperature=0.2)

    evidence_adapter = EvidenceAdapter()
    retrieval_pipeline = RetrievalPipeline(
        KnowledgeBaseClient(runtime_settings),
        evidence_adapter,
        runtime_settings,
    )

    classifier = QuestionClassifier(structured_model, runtime_settings)
    context_resolver = ContextResolver(QuestionRewriter(), structured_model, runtime_settings)
    router = WorkflowRouter(runtime_settings)
    general_workflow = GeneralWorkflow(answer_model)
    policy_assessment = PolicyAssessmentChain(structured_model, runtime_settings)
    policy_query_parser = PolicyQueryParser(structured_model, runtime_settings)
    policy_workflow = PolicyWorkflow(
        retrieval_pipeline,
        policy_query_parser,
        policy_assessment,
        answer_model,
        evidence_adapter,
        runtime_settings,
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
    checkpoint_store = SQLiteCheckpointStore(runtime_settings.checkpoint_path)
    data_workflows = {
        profile.category: DataDomainWorkflow(
            profile,
            profile_map,
            planner,
            answer_model,
            evidence_adapter,
            runtime_settings,
            retrieval_pipeline,
            sql_gateway,
            clients,
            analyzer,
            checkpoint_store,
        )
        for profile in profiles
    }

    dependencies = ApplicationDependencies(
        settings=runtime_settings,
        context_resolver=context_resolver,
        classifier=classifier,
        router=router,
        general_workflow=general_workflow,
        policy_workflow=policy_workflow,
        data_workflows=data_workflows,
    )
    return TenderQAApplication(dependencies)
