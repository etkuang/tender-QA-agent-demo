# PROJECT_STRUCTURE.md

## 1) 项目定位

本仓库是一个四进程招投标智能问答系统：

- `knowledge_base_layer`：独立知识库服务，负责 PDF 解析、Embedding、Milvus 检索和索引发布。
- `agent_layer`：负责问题理解、分类、工作流编排、证据评估和答案生成。
- `app_backend_layer`：负责会话存储和面向前端的 API 网关。
- `app_frontend_layer`：提供 Streamlit 会话界面。

Agent 层覆盖政策、招投标、舆情、公司、价格、产品和其他问题七个分类值，其中前六类对应六类专业问答能力。

## 2) 顶层结构

- `main.py`：按顺序启动 Knowledge Base、Agent、Backend 和 Frontend 四个服务。
- `knowledge_base_layer/`：知识检索 HTTP 契约、Milvus、Embedding、PDF 解析和索引发布。
- `agent_layer/`：分类、上下文解析、工作流、知识库客户端、SQL、安全网关和评测。
- `app_backend_layer/`：后端 API、会话数据库、日志和配置。
- `app_frontend_layer/`：前端 UI 和交互组件。
- `common/`：共享 request ID 日志上下文和日志器。
- `history/`：Backend 的 SQLite 会话数据库运行目录。
- `pyproject.toml`、`uv.lock`：项目依赖和锁文件。
- `招投标六类智能问答系统技术方案报告.docx`：Agent 架构的技术依据。

## 3) 启动链路

`main.py` 使用以下入口启动四个服务：

- Knowledge Base：`uvicorn knowledge_base_layer.api:app`
- Agent：`uvicorn agent_layer.api:app`
- Backend：`uvicorn app_backend_layer.api:app`
- Frontend：`streamlit run app_frontend_layer/app.py`

目标调用链为：Frontend -> Backend -> Agent -> Knowledge Base -> Milvus。

Agent 对 Backend 暴露 `POST /chat/stream` NDJSON 事件流。Knowledge Base 对 Agent 暴露 `POST /search`，并通过 `X-Request-ID` 传递请求关联标识。`main.py` 动态分配 Knowledge Base 端口，并通过 `KNOWLEDGE_BASE_URL` 注入 Agent 进程。

## 4) Agent 层

### 4.1 应用和协议

- `api.py`：FastAPI 入口、请求取消检测、异常到稳定错误码的映射、NDJSON 输出。
- `app.py`：`TenderQAApplication` 应用门面，串联上下文解析、分类、路由、工作流、检查点和指标。
- `bootstrap.py`：组合根，构造模型、知识库 HTTP 客户端、工作流和应用依赖。
- `config.py`：LLM、知识库 URL、路由阈值、SQL 和检查点配置。
- `schemas.py`：分类、实体、证据、研究计划、任务结果、事件和指标等结构化模型。
- `errors.py`：稳定错误类型及外部模型/网络异常映射。
- `checkpoint.py`：独立于聊天历史的 SQLite 执行检查点。

Agent 内部事件包括：`route`、`progress`、`reasoning_summary`、`source`、`assistant_delta`、`final`、`error`。`api.py` 在 HTTP 边界将内部事件整理为 Backend 当前契约支持的 `reasoning`、`assistant`、`error`。

### 4.2 分类和上下文

- `classification/chain.py`：基于模型结构化输出的问题分类器。
- `classification/prompts.py`：六类专业问题和 `other` 的分类约束。
- `context.py`：解析指代、历史实体和当前问题，生成规范化独立问题。

分类结果包含主分类、次分类、置信度、理由、是否需要新鲜数据和结构化实体。低置信度请求进入澄清路由。

### 4.3 工作流

- `workflows/router.py`：按分类和置信度选择专业工作流、澄清或通用回答。
- `workflows/policy.py`：解析政策查询、调用知识库、评估证据充分性，并在配置适配器时补充官方互联网来源。
- `workflows/data_domain.py`：五类数据域共享的计划、DAG 调度、并发执行、证据汇总和回答流程。
- `workflows/analysis.py`：对结构化结果执行确定性统计，模型只负责解释。
- `workflows/common.py`：领域配置、引用标记校验、证据格式化和安全输出辅助。
- `workflows/general.py`：`other` 类直接回答，不调用专业数据工具。

### 4.4 检索边界

- `retrieval/client.py`：异步调用 Knowledge Base `POST /search`。
- `retrieval/pipeline.py`：将 Agent 的政策或领域查询转换为知识库请求，并把返回结果适配为 Agent 证据。
- `retrieval/adapter.py`：将知识库、网站和 SQL 结果统一转换为 `Evidence`。
- `retrieval/chinese_number.py`、`rewrite.py`：条号解析和问题改写辅助。

Agent 不再直接持有 Milvus 凭据、Embedding 模型、向量检索、融合、父文档扩展或索引发布逻辑。

### 4.5 五类数据域

- `domains/tender.py`
- `domains/public_opinion.py`
- `domains/company.py`
- `domains/price.py`
- `domains/product.py`

五类数据域当前未配置本地知识索引，`knowledge_index` 均为 `None`。SQL 和网站能力保留，由部署环境注入实际执行器或适配器。

### 4.6 SQL 和外部数据

- `sql/validator.py`：使用 `sqlglot` AST 校验只读 SQL、视图白名单和强制 `LIMIT`。
- `sql/gateway.py`：结构化 SQL 生成、校验、只读执行、超时和审计事件。
- `sql/catalog.py`、`sql/schemas.py`：五类数据域默认视图目录和 SQL 模型。
- `adapters/`：政策互联网及五类网站数据源的显式抽象接口。

具体数据库连接、真实视图 schema、网站端点、凭据和合规策略由部署环境注入。

## 5) Knowledge Base 层

### 5.1 HTTP 契约和服务

- `schemas.py`：`KnowledgeSearchRequest`、`KnowledgeHit` 和 `KnowledgeSearchResponse`。这些协议模型按当前设计归属 Knowledge Base 层，而不是 `common/`。
- `api.py`：FastAPI 健康检查和搜索接口。
- `service.py`：逻辑索引解析、精确条款查询、混合检索、父文档扩展和可选重排。
- `bootstrap.py`：构造 Milvus、Embedding、融合器、检索器和服务。
- `config.py`：Milvus、Embedding、检索、PDF 和逻辑索引配置。

### 5.2 检索和 Milvus

- `retrieval/milvus.py`：Milvus REST v2 的检索、查询、写入、建集合和 alias 切换实现。
- `retrieval/retriever.py`：Dense 与 BM25 召回及政策标量过滤。
- `retrieval/fusion.py`：RRF 或加权融合。
- `retrieval/parent_context.py`：child 命中后的 parent 回查。
- `retrieval/reranker.py`：可注入重排器协议。
- `retrieval/store.py`：向量存储协议。
- `embeddings.py`：延迟加载本地 Sentence Transformer 模型。

### 5.3 PDF 重建和发布

- `data/pdf_sources.json`：两份政策 PDF 的路径、版式、来源类型、快照日期和页码范围。
- `data/pdf/`：政策法规汇编和实务解读原始 PDF。
- `ingestion/pdf.py`：单栏或双栏 PDF 文本提取、法条识别、父子切块和来源元数据生成。
- `ingestion/normalizer.py`：统一字段、稳定 ID 和质量问题模型。
- `ingestion/pipeline.py`：Embedding、版本集合写入和 alias 发布。
- `rebuild.py`：从 PDF manifest 重建政策集合的命令行入口。

当前可重建的本地知识索引只有逻辑索引 `policy`，稳定 alias 默认为 `tender_qa_policy`。旧 Chroma 数据库和 Chroma-to-Milvus 迁移路径已删除。

## 6) Backend 层

- `api.py`：服务入口和会话存储生命周期。
- `api_routes/chat.py`：加载历史、调用 Agent、透传流并写回历史。
- `api_routes/sessions.py`：会话列表、消息读取和删除。
- `history/history_db.py`：SQLite 会话与消息持久化。

Backend 通过 `AGENT_BASE_URL` 调用 Agent。

## 7) Frontend 层

- `app.py`：Streamlit 入口。
- `api_client.py`：Frontend 到 Backend 的 API 客户端。
- `components/chat_view.py`：聊天展示和流式消费。
- `components/sidebar.py`：会话管理和设置。

Frontend 通过 `BACKEND_URL` 调用 Backend。

## 8) 运行约束

1. 项目使用 namespace package，运行时需要把项目根目录加入 `PYTHONPATH`。
2. 在线知识检索需要可访问的 Milvus 服务及已发布的 `tender_qa_policy` alias。
3. Embedding 维度必须与 Milvus collection schema 一致，默认维度为 768。
4. SQL 能力需要安装 `sqlglot`，并注入只读执行器和审计落地实现。
5. 网站和政策互联网能力需要注入具体适配器；未配置时不会虚构外部检索结果。
6. 两份政策 PDF 是截至 2022 年的来源快照，不能直接视为当前有效法律全集。
