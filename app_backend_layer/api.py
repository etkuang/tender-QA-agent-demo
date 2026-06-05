# coding: utf-8
# @Author: Wang Qingkang

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app_backend_layer.api_routes.chat import router as chat_router
from app_backend_layer.api_routes.sessions import router as sessions_router
from common.logging import reset_request_id, set_request_id
from app_backend_layer.history.history_db import HistoryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = HistoryManager()
    await db.connect()
    await db.initialize_db()
    app.state.db = db
    yield
    await db.disconnect()


app = FastAPI(title="AI Inference Engine Backend", lifespan=lifespan)


@app.middleware("http")
async def inject_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(sessions_router)
app.include_router(chat_router)
