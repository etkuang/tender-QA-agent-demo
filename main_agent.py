"""主流程：LLM意图分类 + 简单问题RAG + 复杂问题ReAct Agent（配置化版本，带会话记忆）"""
import time
import uuid
import re
from typing import Optional, List, Dict
from fastapi import FastAPI

from pydantic import BaseModel

from app.core.retriever import HybridRetriever
from app.core.generator import LLMGenerator
from app.core.intent_router import IntentRouter
from app.core.session_manager import SessionManager
from app.agent.react_agent import ReActAgent
from app.tools.regulation_tools import summarize
from config import settings
from app.utils.question_rewriter import question_rewriter


# ========== 请求/响应模型 ==========
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    processing_time: float
    session_id: str
    route: str  # greeting / rejected / rag / agent
    sources: List[Dict] = []  # ← 添加这行

# ========== 指代词检测 ==========
REFERENTIAL_PATTERNS = re.compile(
    r'(它|他|她|这个|那个|上面|刚才|那|这|前面|上述|该|其|如此|这样|那样|此类|这些|那些)'
)


def has_referential_word(question: str) -> bool:
    """检测问题是否包含指代词"""
    return bool(REFERENTIAL_PATTERNS.search(question))


# ========== 主控制器（带会话记忆）==========
class BidAssistant:
    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator):
        self.retriever = retriever
        self.llm = llm
        self.session_manager = SessionManager()
        self.intent_router = IntentRouter(llm)
        self.agent = ReActAgent(
            retriever,
            llm,
            max_steps=settings.react_max_steps,
            session_manager=self.session_manager
        )

    async def _rewrite_question(self, question: str, session_id: str) -> str:
        """
        问题改写：先做规则预处理，再做指代消解（LLM），最后规则润色
        """
        # 第一步：规则预处理（口语转书面语、冗余精简、同义词）
        # 注意：这一步在指代消解之前做，可以让指代消解更准确
        preprocessed = question
        if question_rewriter.is_enabled():
            preprocessed = question_rewriter.rewrite(question)
            print(f"   ✍️ 规则预处理: {question} → {preprocessed}")  # ← 加这行看日志
        # 第二步：指代消解（使用 LLM，仅当有指代词时）
        rewritten = preprocessed
        if settings.enable_question_rewrite and has_referential_word(preprocessed):
            # 获取历史对话
            history = self.session_manager.get_recent_dialogues(session_id, max_pairs=2)
            if history:
                prompt = f"""将用户问题中的指代词（如"它"、"那"、"这个"）替换为具体内容。

    历史对话：
    {history}

    用户问题：{preprocessed}

    规则：
    1. 只输出重写后的问题，不要输出任何解释
    2. 如果没有指代词或无法确定指代对象，原样输出原问题
    3. 输出不要包含引号

    重写后的问题："""

                try:
                    rewritten = await self.llm.quick_generate(prompt, max_tokens=200)
                    rewritten = rewritten.strip().strip('"').strip("'")

                    if len(rewritten) < 3:
                        rewritten = preprocessed
                    else:
                        print(f"   📝 指代消解: {preprocessed[:50]}... → {rewritten[:50]}...")
                except Exception as e:
                    print(f"   ⚠️ 指代消解失败: {e}")
                    rewritten = preprocessed

        # 第三步：规则润色（对指代消解后的结果做最后清理）
        final_question = rewritten
        if question_rewriter.is_enabled() and rewritten != preprocessed:
            # 只做轻量清理，避免重复替换
            final_question = question_rewriter._normalize_punctuation(rewritten)
            final_question = re.sub(r'\s+', ' ', final_question).strip()

        return final_question

    async def ask(self, question: str, session_id: str) -> tuple:
        # 临时调试：打印问题改写前后的对比
        from app.utils.question_rewriter import question_rewriter
        rewritten_test = question_rewriter.rewrite(question)
        if question != rewritten_test:
            print(f"\n🔧 [调试] 问题改写应该生效:")
            print(f"   原始: {question}")
            print(f"   改写: {rewritten_test}")
        else:
            print(f"\n🔧 [调试] 问题改写未生效或无需改写: {question}")
        start_time = time.time()

        # 记录用户消息
        self.session_manager.add_user_message(session_id, question)

        # 问题重写
        rewritten_question = await self._rewrite_question(question, session_id)
        final_question = rewritten_question

        # 意图路由
        intent = await self.intent_router.route(
            question=final_question,
            session_id=session_id,
            session_manager=self.session_manager
        )
        print(f"\n📋 [意图] type={intent['type']}, complexity={intent['complexity']}")

        sources = []  # ← 初始化 sources

        # 快速回复
        if intent["type"] == "quick_response":
            print(f"👋 [路由] 快速回复")
            route = "greeting"
            answer = intent.get("response", "您好！请问有什么可以帮您？")
            self.session_manager.add_ai_message(session_id, answer)
            elapsed = time.time() - start_time
            return answer, route, elapsed, sources  # ← 添加 sources

        # 无关问题
        if intent["type"] == "unrelated":
            print(f"🚫 [路由] 无关问题，直接拒绝")
            answer = settings.unrelated_response
            route = "rejected"
            self.session_manager.add_ai_message(session_id, answer)
            elapsed = time.time() - start_time
            return answer, route, elapsed, sources  # ← 添加 sources

        # 正常流程
        if intent["complexity"] == "single_step":
            print(f"⚡ [路由] 简单问题 → 直接RAG")
            results = self.retriever.search(final_question, "regulations", top_k=settings.top_k)

            # ← 构建 sources
            sources = []
            for r in results[:settings.top_k]:
                sources.append({
                    "title": r.get("data", {}).get("doc_title", r.get("data", {}).get("source", "未知")),
                    "content_preview": r.get("text", "")[:200],
                    "score": r.get("score", 0)
                })

            if not results:
                answer = settings.no_results_response
            else:
                answer = await summarize(results, final_question, self.llm)
            route = "rag"
        else:
            print(f"🧠 [路由] 复杂问题 → ReAct Agent模式")
            answer = await self.agent.run(final_question, session_id)
            route = "agent"
            # Agent 模式暂时不返回 sources（可后续扩展）

        # 记录 AI 回复
        self.session_manager.add_ai_message(session_id, answer)

        elapsed = time.time() - start_time
        return answer, route, elapsed, sources  # ← 添加 sources


# ========== FastAPI ==========
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description
)



retriever = HybridRetriever()
llm = LLMGenerator()
assistant = BidAssistant(retriever, llm)


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
    answer, route, elapsed, sources = await assistant.ask(req.question, session_id)  # ← 接收 sources

    return AskResponse(
        answer=answer,
        processing_time=elapsed,
        session_id=session_id,
        route=route,
        sources=sources  # ← 添加 sources
    )


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "stats": retriever.get_stats()}


@app.delete("/api/v1/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话（用于测试或手动重置）"""
    assistant.session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/")
async def root():
    return {
        "service": settings.app_title,
        "version": settings.app_version,
        "architecture": {
            "intent": "LLM分类 (type/complexity) + 1轮历史感知",
            "simple": "直接RAG → summarize",
            "complex": f"ReAct Agent (Thought→Action→Observation循环, {settings.react_max_steps}步, 5轮历史)",
            "memory": "SQLite会话记忆 (LangChain)",
            "rewrite": "指代词自动重写"
        }
    }


@app.get("/api/v1/config/info")
async def config_info():
    """查看当前配置信息（调试用）"""
    return {
        "fusion_strategy": settings.fusion_strategy,
        "react_max_steps": settings.react_max_steps,
        "top_k": settings.top_k,
        "vector_recall": settings.vector_recall,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }


@app.get("/api/v1/session/stats")
async def session_stats():
    """获取会话统计信息"""
    return assistant.session_manager.get_stats()


@app.get("/api/v1/session/{session_id}/info")
async def session_info(session_id: str):
    """获取指定会话的详细信息"""
    return assistant.session_manager.get_session_info(session_id)


@app.post("/api/v1/session/cleanup")
async def cleanup_expired_sessions():
    """手动触发清理过期会话"""
    cleaned = assistant.session_manager.cleanup_expired_sessions()
    return {"cleaned": cleaned, "active": assistant.session_manager.get_stats()["active_sessions"]}


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 启动服务: {settings.host}:{settings.port}")
    print(f"📋 服务名称: {settings.app_title}")
    print(f"🔧 融合策略: {settings.fusion_strategy}")
    print(f"🧠 ReAct最大步数: {settings.react_max_steps}")
    print(f"💾 会话记忆存储: {settings.sqlite_db_path}")

    #启动时清理一次过期会话
    cleaned = assistant.session_manager.cleanup_expired_sessions()
    print(
        f"📊 启动时清理完成，清理了 {cleaned} 个过期会话，当前活跃会话: {assistant.session_manager.get_stats()['active_sessions']}")

    uvicorn.run(
        "main_agent:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )