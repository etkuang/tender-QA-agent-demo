# coding: utf-8
# @Author: Wang Qingkang

from knowledge_base_layer.config import Settings, settings
from knowledge_base_layer.embeddings import LocalEmbeddingGateway
from knowledge_base_layer.retrieval.fusion import HybridFusion
from knowledge_base_layer.retrieval.milvus import MilvusStore
from knowledge_base_layer.retrieval.parent_context import ParentContextExpander
from knowledge_base_layer.retrieval.retriever import HybridRetriever
from knowledge_base_layer.service import KnowledgeBaseService


def build_service(runtime_settings: Settings = settings) -> KnowledgeBaseService:
    store = MilvusStore(runtime_settings)
    embeddings = LocalEmbeddingGateway(runtime_settings.embedding_model_name_or_path)
    fusion = HybridFusion(runtime_settings)
    retriever = HybridRetriever(store, embeddings, fusion, runtime_settings)
    return KnowledgeBaseService(
        retriever,
        ParentContextExpander(store),
        runtime_settings,
    )