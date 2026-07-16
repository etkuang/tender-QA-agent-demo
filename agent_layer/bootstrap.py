# coding: utf-8

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.config import Settings, settings
from agent_layer.conversation.quick_classifier import QuickResponseClassifier
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.models import ModelFactory
from agent_layer.question_decomposition.chain import QuestionDecomposer
from agent_layer.workflows.factory import build_child_task_workflows
from agent_layer.workflows.general import CompositeAnswerWorkflow


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

    dependencies = ApplicationDependencies(
        settings=runtime_settings,
        quick_classifier=QuickResponseClassifier(structured_model, runtime_settings),
        decomposer=QuestionDecomposer(structured_model, runtime_settings),
        child_task_workflows=build_child_task_workflows(
            structured_model,
            answer_model,
            runtime_settings,
            checkpoint_runtime,
            sql_gateway,
            website_clients,
            policy_internet_client,
            general_internet_client,
        ),
        composite_workflow=CompositeAnswerWorkflow(answer_model, runtime_settings),
        checkpoint_runtime=checkpoint_runtime,
    )
    return TenderQAApplication(dependencies)
