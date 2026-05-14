"""API路由"""

import time
from fastapi import APIRouter, Request

from app.api.schemas import AskRequest, AskResponse, SourceInfo, HealthResponse
from app.core.session_manager import SessionManager
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["rag"])
session_manager = SessionManager()


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, req: AskRequest):
    """问答接口"""
    start_time = time.time()

    # 1. 会话管理
    session_id, is_new = await session_manager.get_or_create_session(req.session_id)
    history = await session_manager.get_history(session_id, last_n=settings.max_history)
    rewritten_question = await session_manager.rewrite_query_with_context(
        session_id, req.question
    )

    # 2. 意图路由（纯关键词）
    route = request.app.state.router.route(rewritten_question)

    # 3. 混合检索
    results = request.app.state.retriever.search(
        query=rewritten_question,
        collection=route["collection"],
        top_k=req.top_k
    )

    # 4. LLM生成
    answer = await request.app.state.generator.generate_with_history(
        query=rewritten_question,
        context=results,
        collection=route["collection"],
        history=history
    )

    # 5. 提取实体
    entities = await session_manager.extract_entities(rewritten_question, answer, results)

    # 6. 保存会话
    await session_manager.add_turn(session_id, req.question, answer, entities)

    # 7. 构建返回
    # sources = []
    # for r in results[:req.top_k]:
    #     data = r.get("data", {})
    #     sources.append(SourceInfo(
    #         title=data.get("title", data.get("name", "")),
    #         project_name=data.get("project_name", ""),
    #         winner=data.get("winner", ""),
    #         winner_amount=data.get("winner_amount"),
    #         content_preview=r.get("text", "")[:200],
    #         source_type=route["collection"],
    #         score=r.get("score", 0)
    #     ))
    sources = []
    for r in results[:req.top_k]:
        data = r.get("data", {})

        # 适配新的元数据结构
        if route["collection"] == "bids":
            sources.append(SourceInfo(
                title=data.get("project_name", data.get("title", "")),
                project_name=data.get("project_name", ""),
                winner=data.get("winner", ""),
                winner_amount=data.get("winner_amount"),
                content_preview=r.get("text", "")[:200],
                source_type=route["collection"],
                score=r.get("score", 0)
            ))
        else:
            # regulations 和 prices 保持原样
            sources.append(SourceInfo(
                title=data.get("title", data.get("name", "")),
                project_name=data.get("project_name", ""),
                winner=data.get("winner", ""),
                winner_amount=data.get("winner_amount"),
                content_preview=r.get("text", "")[:200],
                source_type=route["collection"],
                score=r.get("score", 0)
            ))

    return AskResponse(
        answer=answer,
        sources=sources,
        processing_time=time.time() - start_time,
        session_id=session_id
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    await session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """健康检查"""
    return HealthResponse(
        status="ok",
        collections=request.app.state.retriever.get_stats(),
        embedding_model=settings.embedding_model
    )
