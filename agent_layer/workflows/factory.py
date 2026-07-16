# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.language_models import BaseChatModel

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.checkpoint import LangGraphCheckpointRuntime
from agent_layer.config import Settings
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.schemas import Category
from agent_layer.workflows.child_task import GeneralChildTaskWorkflow
from agent_layer.workflows.policy import PolicyAssessmentChain, PolicyQueryParser
from agent_layer.workflows.policy_graph import PolicyWorkflow
from agent_layer.workflows.profiles import build_general_child_task_profiles
from agent_layer.workflows.self_rag import PolicySelfRAGPlugin
from agent_layer.workflows.tool_pool import initialize_child_task_tool_pool


def build_child_task_workflows(
    structured_model: BaseChatModel,
    answer_model: BaseChatModel,
    runtime_settings: Settings,
    checkpoint_runtime: LangGraphCheckpointRuntime,
    sql_gateway: SQLGateway | None,
    website_clients: dict[str, WebsiteSearchClient] | None,
    policy_internet_client: WebsiteSearchClient | None,
    general_internet_client: WebsiteSearchClient | None,
) -> dict[Category, GeneralChildTaskWorkflow | PolicyWorkflow]:
    initialize_child_task_tool_pool(
        structured_model,
        runtime_settings,
        sql_gateway,
        website_clients,
        policy_internet_client,
        general_internet_client,
    )
    general_workflows = {
        profile.category: GeneralChildTaskWorkflow(
            profile,
            structured_model,
            answer_model,
            runtime_settings,
        )
        for profile in build_general_child_task_profiles()
    }
    return {
        **general_workflows,
        Category.POLICY: PolicyWorkflow(
            PolicyQueryParser(structured_model, runtime_settings),
            PolicyAssessmentChain(structured_model, runtime_settings),
            answer_model,
            runtime_settings,
            checkpoint_runtime,
            PolicySelfRAGPlugin(),
        ),
    }