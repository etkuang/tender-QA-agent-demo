# PROJECT_STRUCTURE.md

## 1) 项目定位
本仓库是一个三层架构的招投标法规问答系统：
- `agent_layer`：意图路由、检索融合、答案生成（含 ReAct）。
- `app_backend_layer`：会话存储与统一流式 API 网关。
- `app_frontend_layer`：Streamlit 会话界面。

## 2) 顶层结构（当前）
- `main.py`：一键启动三层服务，动态分配端口并注入环境变量。
- `agent_layer/`：Agent 核心能力与离线脚本。
- `app_backend_layer/`：后端 API、会话数据库、日志与配置。
- `app_frontend_layer/`：前端 UI 与交互组件。
- `reference/`：参考工程与历史向量数据来源目录（可作为迁移来源，不应作为运行时依赖）。
- `pyproject.toml`：Python 依赖与项目信息（使用 `uv`）。
- `uv.lock`：依赖锁定文件。
- `workflow.md`：流程草图（Mermaid）。
- `codex_change_proposal.txt`：提案记录文件。

## 3) 启动与运行入口

### 3.1 一键启动入口
- `main.py`
  - 启动前设置：`PYTHONPATH=<project_root>`。
  - 启动 Agent：`uvicorn agent_layer.interfaces.http.app:app`。
  - 启动 Backend：`uvicorn app_backend_layer.api:app`。
  - 启动 Frontend：`streamlit run app_frontend_layer/app.py`。
  - 注入：
    - `AGENT_BASE_URL=http://127.0.0.1:{agent_port}/v1`
    - `BACKEND_URL=http://127.0.0.1:{backend_port}`

### 3.2 运行时调用链
1. Frontend 调 Backend：`POST /chat/stream`（NDJSON 流）。
2. Backend 通过 `ChatOpenAI(base_url=AGENT_BASE_URL)` 调 Agent 的 OpenAI 兼容接口。
3. Agent 执行：问题改写 -> 意图路由 -> 检索融合 -> 总结回答 / ReAct。

## 4) Agent 层（`agent_layer`）

### 4.1 接口层（HTTP）
- `interfaces/http/app.py`：创建 FastAPI 应用，初始化 `HybridRetriever`、`LLMGenerator`、`AskService`。
- `interfaces/http/routes.py`：
  - `POST /api/v1/ask`
  - `GET /v1/models`
  - `POST /v1/chat/completions`（支持 SSE）
- `interfaces/http/schemas.py`：Ask/OpenAI 协议数据模型。

### 4.2 应用层（Application）
- `application/services/ask_service.py`：主编排（改写 -> 意图 -> 单步 RAG 或 ReAct）。
- `application/services/react_agent_service.py`：ReAct 循环执行（Thought/Action/Observation）。
- `application/services/rewrite_service.py`：问题改写入口。
- `application/session_manager.py`：会话管理（TTL、截断、清理线程）。

### 4.3 领域层（Domain）
- `domain/intent/router.py`：规则 + LLM 的意图与复杂度判定。
- `domain/retrieval/retriever.py`：混合检索主入口。
  - 支持 Weighted/RRF。
  - 支持法条聚合。
  - `search_article_exact` 兼容两类元数据：
    - 本项目旧字段：`article_num/type`
    - reference 风格字段：`article_id/law_name/chunk_type`
- `domain/retrieval/fusion_policy.py`：Weighted 融合策略（归一化、加权、Boost/Penalty）。
- `domain/retrieval/question_rewriter.py`：规则改写器（口语转书面、冗余清理、同义词扩展）。
- `domain/retrieval/chinese_number.py`：中文数字与法条编号转换。
- `domain/tools/regulation_tools.py`：法规检索、法条查询、摘要工具。
- `domain/tools/agent_tools.py`：ReAct 工具封装。

### 4.4 基础设施层（Infrastructure）
- `infrastructure/config/agent_settings.py`：统一 Settings（环境变量 + `config.yaml`）。
  - 当前默认 embedding 模型：`moka-ai/m3e-base`（768 维）。
  - 向量库路径：`agent_layer/data/databases/chroma_db`。
- `infrastructure/config/config_loader.py`：YAML 配置加载器。
- `infrastructure/config/config.yaml`：检索、意图、PDF 切分、问题改写等配置。
- `infrastructure/llm/client.py`：外部 LLM HTTP 调用封装。
- `infrastructure/embedding/service.py`：`SentenceTransformer` 嵌入服务单例（支持 `embed_batch`/`embed_query`）。
- `infrastructure/vector/chroma_store.py`：ChromaDB 适配。
  - 检索使用 `query_texts`。
  - `get_collection` 对已有集合也绑定 embedding function。
- `infrastructure/cache/bm25_cache.py`：BM25 全局缓存单例。

### 4.5 脚本与数据
- `scripts/init_pdf.py`：法规 PDF 建库脚本（已迁移为 reference 风格）。
  - `law_article`：Parent-Child 结构化切块。
  - `sliding`：滑动窗口切块。
  - 法条结构链路：`LegalStructureParser -> ParentChunkBuilder -> ChildChunkBuilder`。
- `scripts/eval_recall.py`：检索效果评估脚本。
- `scripts/e2e_eval.py`：端到端评估脚本。
- `scripts/compare_fusion.py`：融合策略对比脚本。
- `scripts/debug_failed_queries.py`：失败查询诊断脚本。
- `data/`：
  - `pdf/`：法规 PDF 语料。
  - `databases/chroma_db/`：当前运行使用的 Chroma 向量库。
  - `eval_set_25.json`、`eval_set_200.json`：评估集。

## 5) Backend 层（`app_backend_layer`）

### 5.1 服务入口
- `api.py`
  - `lifespan` 中初始化 `HistoryManager`（aiosqlite）。
  - 中间件注入 `X-Request-ID`。
  - 挂载 `sessions` 与 `chat` 路由。

### 5.2 API 路由
- `api_routes/chat.py`
  - `POST /chat/stream`
  - 加载会话历史，调用 Agent，透传 NDJSON 流并写回历史。
- `api_routes/sessions.py`
  - `GET /sessions/get_list`
  - `GET /sessions/{session_id}/messages`
  - `DELETE /sessions/{session_id}`

### 5.3 存储与通用模块
- `history/history_db.py`：SQLite 会话与消息持久化。
- `core/config.py`：Backend/Frontend 共享配置（URL、超时、DB 路径）。
- `core/logger.py`：带 request_id 的日志器。
- `models/schemas.py`：请求与响应模型。

## 6) Frontend 层（`app_frontend_layer`）

### 6.1 入口
- `app.py`
  - 初始化会话状态与参数配置。
  - 渲染侧边栏和聊天区。

### 6.2 组件
- `components/chat_view.py`：对话展示、输入提交、流式消费 NDJSON。
- `components/sidebar.py`：会话列表、切换、删除、新建与设置入口。
- `components/settings.py`：温度/Top-P/Max Tokens 配置。
- `config.py`：读取/保存 `config.json`。

## 7) 路径治理与配置边界（关键）
- Agent 路径统一由 `agent_layer/infrastructure/config/agent_settings.py` 解析输出。
- Backend 路径统一由 `app_backend_layer/core/config.py` 管理。
- Frontend 本地配置路径由 `app_frontend_layer/config.py` 管理。
- 运行时不应依赖 `reference/` 目录；reference 仅可作为迁移来源。

## 8) 当前实现注意点
1. 工程使用 namespace package 机制（无 `__init__.py`），运行时需保证项目根目录在 `PYTHONPATH`。
2. Agent 脚本建议统一通过 `python -m agent_layer.scripts.<script_name>` 调用。
3. `init_pdf.py` 重建 `regulations` 集合前会先清空该集合。
4. 若重建向量库，embedding 模型维度必须与目标库一致（当前默认 768 维）。
