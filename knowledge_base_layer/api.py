# coding: utf-8
# @Author: Wang Qingkang

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException

from common.api_contracts.knowledge_base_api import KnowledgeSearchRequest, KnowledgeSearchResponse
from common.logger import reset_request_id, set_request_id
from knowledge_base_layer.bootstrap import build_service
from knowledge_base_layer.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.knowledge_base = build_service()
    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=KnowledgeSearchResponse)
async def search(
    request: KnowledgeSearchRequest,
    x_request_id: str = Header(alias="X-Request-ID"),
) -> KnowledgeSearchResponse:
    token = set_request_id(x_request_id)
    try:
        return await asyncio.to_thread(app.state.knowledge_base.search, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown knowledge index: {exc.args[0]}") from exc
    finally:
        reset_request_id(token)