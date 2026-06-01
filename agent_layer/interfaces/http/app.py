# coding: utf-8
# @Author: Wang Qingkang

from fastapi import FastAPI

from agent_layer.application.services.ask_service import AskService
from agent_layer.interfaces.http.routes import bind_routes
from agent_layer.infrastructure.config.agent_settings import settings
from agent_layer.infrastructure.llm.client import LLMGenerator
from agent_layer.domain.retrieval.retriever import HybridRetriever

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
)

retriever = HybridRetriever()
llm = LLMGenerator()
ask_service = AskService(retriever, llm)
app.include_router(bind_routes(ask_service))