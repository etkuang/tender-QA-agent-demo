# coding: utf-8
# @Author: Wang Qingkang

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException

from app_backend_layer.api_routes.chat import router as chat_router
from app_backend_layer.api_routes.sessions import router as sessions_router
from common.logger import reset_request_id, set_request_id
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
    """ Require an X-Request-ID header and bind it to the backend logging context.
        Reset the context after the route finishes so request ids do not leak across requests."""
    request_id = request.headers.get("X-Request-ID")
    if request_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Request-ID header")
    token = set_request_id(request_id)
    try:
        return await call_next(request)
    finally:
        reset_request_id(token)


app.include_router(sessions_router)
app.include_router(chat_router)
