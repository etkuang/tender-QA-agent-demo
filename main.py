#!/usr/bin/env python
"""API服务入口"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.retriever import HybridRetriever
from app.core.generator import LLMGenerator
from app.core.router import IntentRouter
from app.storage.redis_client import redis_client
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("招投标智能问答系统启动中...")
    print("=" * 60)

    print("\n[1/5] 连接Redis...")
    await redis_client._get_client()

    print("\n[2/5] 加载意图路由器...")
    app.state.router = IntentRouter()

    print("\n[3/5] 加载混合检索器...")
    app.state.retriever = HybridRetriever()

    print("\n[4/5] 加载LLM生成器...")
    app.state.generator = LLMGenerator()

    print("\n[5/5] 获取统计信息...")
    stats = app.state.retriever.get_stats()

    print("\n" + "=" * 60)
    print("系统启动成功!")
    print(f"  招标库(bids): {stats.get('bids', 0)} 条")
    print(f"  法规库(regulations): {stats.get('regulations', 0)} 条")
    print(f"  价格库(prices): {stats.get('prices', 0)} 条")
    print(f"  API地址: http://{settings.host}:{settings.port}")
    print(f"  Reranker: {'启用' if settings.use_reranker else '禁用'}")
    print(f"  意图路由: 纯关键词匹配")
    print("=" * 60)

    yield

    print("\n系统关闭...")
    await redis_client.close()


app = FastAPI(
    title="招投标智能问答系统",
    description="基于RAG架构的招投标问答服务",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "招投标智能问答系统",
        "version": "4.0.0",
        "features": [
            "纯关键词意图路由（无BERT，100%可控）",
            "Chroma向量数据库",
            "混合检索（向量 + BM25 + jieba分词）",
            "Reranker精排",
            "多轮对话会话管理"
        ],
        "endpoints": [
            {"path": "POST /api/v1/ask", "description": "问答接口"},
            {"path": "GET /api/v1/health", "description": "健康检查"},
            {"path": "DELETE /api/v1/session/{session_id}", "description": "删除会话"}
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
