# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.language_models import BaseChatModel

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.adapters.website_clients import normalize_website_clients
from agent_layer.config import Settings
from agent_layer.data_domain.analysis import DataResultAnalyzer
from agent_layer.data_domain.chains import DataSQLGenerator, DataSQLRepairChain
from agent_layer.data_domain.context.catalog import build_default_data_context_catalog
from agent_layer.data_domain.context.retriever import DataContextRetriever
from agent_layer.data_domain.sql.gateway import SQLGateway
from agent_layer.data_domain.sql.validator import SQLPolicyValidator
from agent_layer.data_domain.web import WebsiteSupplementer
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.client import KnowledgeBaseClient
from agent_layer.retrieval.pipeline import RetrievalPipeline
from agent_layer.workflows.tools import (
    ChildTaskToolRegistry,
    RAGChildTaskTool,
    SQLChildTaskTool,
    WebsiteChildTaskTool,
)


def initialize_child_task_tool_pool(
    structured_model: BaseChatModel,
    runtime_settings: Settings,
    sql_gateway: SQLGateway | None,
    website_clients: dict[str, WebsiteSearchClient] | None,
    policy_internet_client: WebsiteSearchClient | None,
    general_internet_client: WebsiteSearchClient | None,
) -> None:
    clients = normalize_website_clients(
        website_clients,
        policy_internet_client,
        general_internet_client,
    )
    evidence_adapter = EvidenceAdapter()
    retrieval_pipeline = RetrievalPipeline(
        KnowledgeBaseClient(runtime_settings),
        evidence_adapter,
        runtime_settings,
    )
    context_retriever = DataContextRetriever(
        build_default_data_context_catalog(),
        runtime_settings,
    )
    website_supplementer = WebsiteSupplementer(
        evidence_adapter,
        clients,
        runtime_settings,
    )
    ChildTaskToolRegistry.initialize(
        {
            "rag": RAGChildTaskTool(retrieval_pipeline),
            "sql": SQLChildTaskTool(
                DataSQLGenerator(structured_model, runtime_settings),
                DataSQLRepairChain(structured_model, runtime_settings),
                SQLPolicyValidator(),
                sql_gateway,
                context_retriever,
                DataResultAnalyzer(),
                runtime_settings,
            ),
            "website": WebsiteChildTaskTool(website_supplementer),
        }
    )