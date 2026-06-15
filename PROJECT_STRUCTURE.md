# PROJECT_STRUCTURE.md

## 1) 项目定位

本仓库是一个三层招投标智能问答系统：

- `agent_layer`：按照《招投标六类智能问答系统技术方案报告》实现的智能问答核心。
- `app_backend_layer`：会话存储和面向前端的 API 网关。
- `app_frontend_layer`：Streamlit 会话界面。

Agent 层当前覆盖政策、招投标、舆情、公司、价格、产品和其他问题七个分类值，其中前六类对应报告规定的六类专业问答能力。

## 2) 顶层结构

- `main.py`：启动 Agent、Backend 和 Frontend 三个服务。
- `agent_layer/`：分类、上下文解析、工作流、检索、SQL、安全网关、迁移和评测。
- `app_backend_layer/`：后端 API、会话数据库、日志和配置。
- `app_frontend_layer/`：前端 UI 和交互组件。
- `common/`：共享 request ID 日志上下文和日志器。
- `history/`：Backend 的 SQLite 会话数据库运行目录。
- `pyproject.toml`、`uv.lock`：项目依赖和锁文件。
- `招投标六类智能问答系统技术方案报告.docx`：本次 Agent 架构的技术依据。

## 3) 启动链路

`main.py` 使用以下入口启动三层服务：

- Agent：`uvicorn agent_layer.api:app`
- Backend：`uvicorn app_backend_layer.api:app`
- Frontend：`streamlit run app_frontend_layer/app.py`

目标调用链为：Frontend -> Backend -> Agent `POST /chat/stream`。Agent 接口使用 NDJSON 事件流，并要求请求携带 `request_id`。

## 4) Agent 层

### 4.1 应用和协议

- `api.py`：FastAPI 入口、请求取消检测、异常到稳定错误码的映射、NDJSON 输出。
- `app.py`：`TenderQAApplication` 应用门面，串联上下文解析、分类、路由、工作流、检查点和指标。
- `bootstrap.py`：组合根，集中构造模型、Milvus、检索管线、工作流和应用依赖。
- `config.py`：LLM、Embedding、Milvus、路由阈值、检索、SQL 和检查点配置。
- `schemas.py`：分类、实体、证据、研究计划、任务结果、事件和指标等结构化模型。
- `errors.py`：稳定错误类型及外部模型/网络异常映射。
- `checkpoint.py`：独立于聊天历史的 SQLite 执行检查点。

Agent 内部事件包括：`route`、`progress`、`reasoning_summary`、`source`、`assistant_delta`、`final`、`error`。`api.py` 在 HTTP 边界将这些内部事件整理为 Backend 现有契约支持的 `reasoning`、`assistant`、`error`，其中用户可读的处理过程通过 `reasoning` 实时传递。

### 4.2 分类和上下文

- `classification/chain.py`：基于模型结构化输出的问题分类器。
- `classification/prompts.py`：六类专业问题和 `other` 的分类约束。
- `context.py`：解析指代、历史实体和当前问题，生成规范化独立问题。

分类结果包含主分类、次分类、置信度、理由、是否需要新鲜数据和结构化实体。低置信度请求进入澄清路由，不猜测执行专业工具。

### 4.3 工作流

- `workflows/router.py`：按分类和置信度选择专业工作流、澄清或通用回答。
- `workflows/policy.py`：政策确定性流程，本地精确法条和混合检索优先，证据不足时才允许官方互联网补充。
- `workflows/data_domain.py`：五类数据域共享的计划、DAG 调度、并发执行、证据汇总和回答流程。
- `workflows/analysis.py`：对结构化结果执行确定性统计，模型只负责解释。
- `workflows/common.py`：引用标记校验、证据格式化和安全输出辅助。
- `workflows/general.py`：`other` 类直接回答，不调用专业数据工具。

### 4.4 五类数据域

- `domains/tender.py`
- `domains/public_opinion.py`
- `domains/company.py`
- `domains/price.py`
- `domains/product.py`

每个模块构造自己的 `DomainProfile`，声明允许的数据源、SQL 视图、跨域范围、网站适配器和回答要求。`DataDomainWorkflow` 在执行每个任务前再次校验域、来源和视图权限。

### 4.5 检索和 Milvus

- `retrieval/milvus.py`：Milvus REST v2 实现，支持 dense search、BM25 sparse search、query/get、upsert、版本集合创建和 alias 切换。
- `retrieval/retriever.py`：并行执行向量和关键词检索。
- `retrieval/fusion.py`：默认 RRF 融合，可配置等权加权融合。
- `retrieval/parent_context.py`：child 命中后回查 parent，恢复完整法条上下文。
- `retrieval/reranker.py`：可注入重排器接口。
- `retrieval/pipeline.py`：改写、召回、融合、父子扩展、重排和证据适配的完整管线。
- `retrieval/adapter.py`、`store.py`：统一检索文档和证据边界。

Milvus 是目标运行时检索库。旧 Chroma 数据只作为一次性迁移来源，不再是在线检索回退路径。

### 4.6 SQL 和外部数据

- `sql/validator.py`：使用 `sqlglot` AST 校验只读 SQL、视图白名单和强制 `LIMIT`。
- `sql/gateway.py`：结构化 SQL 生成、校验、只读执行、超时和审计事件。
- `sql/catalog.py`、`schemas.py`：五类数据域默认视图目录和 SQL 模型。
- `adapters/`：政策互联网及五类网站数据源的显式抽象接口。

具体数据库连接、真实视图 schema、网站端点、凭据和合规策略由部署环境注入；Agent 层不伪造这些外部资源。

### 4.7 数据治理、迁移和评测

- `ingestion/normalizer.py`：统一字段、稳定 ID、时间和法规元数据质量校验。
- `ingestion/pipeline.py`：版本化写入和 alias 发布流程。
- `migration/chroma_to_milvus.py`：Chroma 到 Milvus 的单向迁移，关键质量问题存在时禁止写入和发布。
- `evaluation/`：分类、检索和 SQL 的离线评测函数。
- `tests/`：协议、路由、上下文、检查点、Milvus 请求和父子检索等单元测试。

当前旧数据预检结果：`regulations` 共 7319 条，其中 5350 条缺失 `validity_status`，会被迁移门禁阻断；`bids` 共 8789 条，通过当前规范化校验。

## 5) Backend 层

- `api.py`：服务入口和会话存储生命周期。
- `api_routes/chat.py`：加载历史、调用 Agent、透传流并写回历史。
- `api_routes/sessions.py`：会话列表、消息读取和删除。
- `history/history_db.py`：SQLite 会话与消息持久化。

Backend 仍属于旧实现范围。本次严格限制为 Agent 层重构，因此 Backend 对新 Agent 事件协议的适配不在本次修改中。

## 6) Frontend 层

- `app.py`：Streamlit 入口。
- `api_client.py`：Frontend 到 Backend 的 API 客户端。
- `components/chat_view.py`：聊天展示和流式消费。
- `components/sidebar.py`：会话管理和设置。

Frontend 仍属于旧实现范围，本次未修改。

## 7) 运行约束

1. 项目使用 namespace package，运行时需要把项目根目录加入 `PYTHONPATH`。
2. 在线检索需要可访问的 Milvus 服务及已发布 alias。
3. Embedding 维度必须与 Milvus collection schema 一致，默认维度为 768。
4. SQL 能力需要安装 `sqlglot`，并注入只读执行器和审计落地实现。
5. 网站和政策互联网能力需要注入具体适配器；未配置时不会虚构外部检索结果。
6. Agent 内部使用结构化事件，HTTP 输出保持 Backend 当前的三类流消息契约。
