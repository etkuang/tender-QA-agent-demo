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
- `pyproject.toml`：Python 依赖与项目信息（使用 `uv`）。
- `uv.lock`：依赖锁定文件。
- `workflow.md`：流程草图（Mermaid）。
- `codex_change_proposal.txt`：提案记录文件。

## 3) 启动与运行入口

### 3.1 一键启动入口
- `main.py`
  - 启动前设置：`PYTHONPATH=<project_root>`。
  - 启动 Agent：`uvicorn agent_layer.api:app`。
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

Agent 层采用第一层扁平结构；只有当单项文件明显过长时，才将该项升级为同名目录继续分层。

### 4.1 顶层模块
- `api.py`：FastAPI 应用与 OpenAI 兼容协议。
  - `POST /api/v1/ask`
  - `GET /v1/models`
  - `POST /v1/chat/completions`（支持 SSE）
- `config.py`：Settings、路径解析、LLM 构造、默认检索/意图配置。
- `state.py`：Ask/OpenAI 协议模型、Intent/Ask 结果模型、运行态 state。
- `prompts.py`：Intent、RAG、Legal Agent system prompt。
- `retrieval/`：检索子包，包含 Chroma、BM25、Weighted/RRF 融合、问题改写、中文法条编号转换、LangChain `Document` adapter。
  - 支持 Weighted/RRF。
  - 支持法条聚合。
  - `search_article_exact` 兼容两类元数据：
    - 本项目旧字段：`article_num/type`
    - reference 风格字段：`article_id/law_name/chunk_type`
- `tools.py`：`search_law`、`get_article` 两个 LangChain `StructuredTool`。
- `chains.py`：意图识别 chain 与单步 RAG chain。
- `agent.py`：统一 runtime，负责问题改写、寒暄/拒答、RAG/Agent 路由，以及 `create_agent` 多步调用。

### 4.2 数据与运行约束
- 默认 embedding 模型：`moka-ai/m3e-base`（768 维）。
- 默认向量库路径：`agent_layer/data/databases/chroma_db`。
- Agent 层不做会话持久化；Backend 会把历史 messages 传入 OpenAI 兼容接口。
- 运行时数据已迁移到 `agent_layer/data/`。

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
- Agent 路径统一由 `agent_layer/config.py` 解析输出。
- Backend 路径统一由 `app_backend_layer/core/config.py` 管理。
- Frontend 本地配置路径由 `app_frontend_layer/config.py` 管理。
- 运行时数据由 `agent_layer/data/` 提供。

## 8) 当前实现注意点
1. 工程使用 namespace package 机制（无 `__init__.py`），运行时需保证项目根目录在 `PYTHONPATH`。
2. `main.py` 使用 `uvicorn agent_layer.api:app` 启动 Agent。
3. 若重建向量库，embedding 模型维度必须与目标库一致（当前默认 768 维）。
