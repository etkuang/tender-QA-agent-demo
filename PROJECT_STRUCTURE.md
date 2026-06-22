# PROJECT_STRUCTURE.md

## 1. Project Purpose

This repository implements a four-process tender QA system:

- `knowledge_base_layer`: local policy knowledge-base service for PDF ingestion, embedding, Milvus retrieval, fusion, parent-context expansion, and index publishing.
- `agent_layer`: question understanding, decomposition, routing, policy Self-RAG orchestration, data-domain workflows, SQL gateway contracts, website adapter contracts, and answer synthesis.
- `app_backend_layer`: backend API gateway and persistent chat-history storage.
- `app_frontend_layer`: Streamlit chat UI.

The runtime request path is:

Frontend -> Backend -> Agent -> Knowledge Base -> Milvus.

The Agent supports policy, tender, public-opinion, company, price, product, other, and unclear task categories. Multi-part questions are decomposed into child tasks and executed through the relevant workflows before final synthesis.

## 2. Top-Level Structure

- `main.py`: starts Knowledge Base, Agent, Backend, and Frontend services in order.
- `knowledge_base_layer/`: policy document ingestion, Milvus retrieval, and `/search` API.
- `agent_layer/`: Agent API, composition root, LangGraph policy workflow, classification, routing, retrieval adapters, SQL contracts, and data-domain workflows.
- `app_backend_layer/`: session and message persistence plus Backend API routes.
- `app_frontend_layer/`: Streamlit frontend and Backend API client.
- `common/`: shared HTTP DTOs and logging utilities.
- `history/`: runtime location for Backend chat-history SQLite data.
- `pyproject.toml` and `uv.lock`: dependencies and lock file.
- `test.py`: standalone proxy-environment diagnostic script, not an automated test suite.

## 3. Service Startup And API Boundaries

`main.py` starts:

- Knowledge Base: `uvicorn knowledge_base_layer.api:app`
- Agent: `uvicorn agent_layer.api:app`
- Backend: `uvicorn app_backend_layer.api:app`
- Frontend: `streamlit run app_frontend_layer/app.py`

Agent exposes `POST /chat/stream` as NDJSON through `agent_layer/api.py`.

Knowledge Base exposes `POST /search` through `knowledge_base_layer/api.py`.

Shared API contracts live in:

- `common/api_contracts/backend_api.py`: Frontend-Backend DTOs.
- `common/api_contracts/agent_api.py`: Backend-Agent DTOs.
- `common/api_contracts/knowledge_base_api.py`: Agent-Knowledge Base DTOs.

`X-Request-ID` is propagated through request-scoped logging.

## 4. Agent Layer

### 4.1 Composition And Lifecycle

- `agent_layer/api.py`: FastAPI entrypoint, NDJSON transport formatting, request cancellation checks, and application lifespan.
- `agent_layer/bootstrap.py`: async composition root. It creates models, clients, workflows, and opens the LangGraph checkpoint runtime.
- `agent_layer/app.py`: `TenderQAApplication`, streaming orchestration, child-task execution, source streaming, and application shutdown cleanup.
- `agent_layer/config.py`: Agent runtime settings for LLM, Knowledge Base, streaming, retrieval budgets, SQL limits, and LangGraph checkpoint path.
- `agent_layer/checkpoint.py`: `LangGraphCheckpointRuntime`, which owns `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` for the Agent process lifetime.

`build_application()` is asynchronous because it opens the LangGraph SQLite checkpointer before constructing the policy graph.

`TenderQAApplication.aclose()` closes the LangGraph checkpoint runtime during FastAPI shutdown.

### 4.2 Question Understanding And Routing

- `classification/prompts.py`: prompt for decomposition-first question understanding.
- `classification/chain.py`: produces `QuestionDecomposition` structured output.
- Conversation context, entities, and standalone question text are resolved by the classification prompt and chain.
- `workflows/router.py`: converts `QuestionDecomposition` into `RoutePlan`.

Routing is no longer confidence-threshold based. It uses explicit child tasks and `Category.UNCLEAR` to decide whether to clarify, answer generally, or execute specialized workflows.

### 4.3 Policy Workflow

Policy execution is split between:

- `workflows/policy.py`: policy prompts and LangChain components:
  - `PolicyQueryParser`
  - `PolicyAssessmentChain`
  - policy answer prompt
- `workflows/policy_graph.py`: LangGraph `PolicyGraphWorkflow`.

`PolicyGraphWorkflow` builds a `StateGraph` with these nodes:

1. `parse_policy`: extract law name, article id, date, and region.
2. `retrieve`: retrieve local policy evidence through the Knowledge Base.
3. `assess`: grade evidence sufficiency with structured output.
4. `policy_internet`: optional official-internet supplement if an injected adapter is configured.
5. `synthesize`: produce the final answer with citations.

Conditional edges route from assessment back to retrieval, to official internet search, or to synthesis. If official internet evidence is added, the graph reassesses the expanded evidence set.

The graph is compiled with the LangGraph SQLite checkpointer. Each policy run uses its `run_id` as the LangGraph `thread_id`, so checkpoints are stored by run.

### 4.4 Self-RAG Helpers

- `workflows/self_rag.py`: evidence merging, follow-up query selection, and final evidence selection.
- `workflows/common.py`: domain profile model, evidence formatting, citation validation, citation building, and evidence ranking.

Self-RAG retrieval budgets are configured by:

- `retrieval_batch_size`
- `max_retrieval_rounds`
- `max_follow_up_queries`
- `evidence_chunk_chars`
- `evidence_context_chars`

### 4.5 Data-Domain Workflows

Data-domain workflows cover:

- `domains/tender.py`
- `domains/public_opinion.py`
- `domains/company.py`
- `domains/price.py`
- `domains/product.py`

The shared workflow implementation is `workflows/data_domain.py`.

It plans research tasks, executes SQL and website jobs, analyzes structured results, gathers evidence, and synthesizes a final answer. SQL and website capabilities are dependency-injected; missing adapters are reported as skipped rather than fabricated.

Data-domain workflows do not currently use checkpoint persistence. They can be migrated independently if resumable data-domain execution is required later.

### 4.6 SQL Boundary

- `sql/gateway.py`: generates SQL candidates, validates them, executes read-only SQL through an injected executor, and records audit events.
- `sql/validator.py`: uses `sqlglot` to parse a single read-only SELECT/CTE/UNION statement, restrict views and columns, block sensitive columns, block dangerous functions, and enforce LIMIT.
- `sql/catalog.py` and `sql/schemas.py`: default view catalog and SQL model schemas.

The current SQL dialect is an internal validator constant:

- `SQL_DIALECT = "sqlite"`

The only remaining SQL runtime row cap is:

- `sql_max_rows`: maximum rows returned by a read-only SQL query and enforced as LIMIT.

The default Agent application still does not construct a real SQL executor. A read-only executor must be injected for SQL-backed data-domain answers.

### 4.7 Website Adapters

- `adapters/base.py`: shared website adapter protocol and base class.
- `adapters/policy_internet.py`, `tender_web.py`, `public_opinion_web.py`, `company_web.py`, `price_web.py`, `product_web.py`: category marker adapters.

The adapter subclasses currently declare only their expected category. Real request construction and response parsing must be implemented by deployment-specific adapters.

For policy internet evidence, the workflow currently trusts `SourceTier.OFFICIAL` from the injected adapter. It no longer performs domain-suffix verification.

### 4.8 Schemas And Events

- `schemas.py`: categories, entities, decomposition, route plans, policy queries, evidence, citations, research plans, tool events, workflow results, stream events, and metrics.
- Internal event types include route, progress, reasoning summary, source, assistant delta, and error.
- `agent_layer/api.py` adapts internal stream events to the Backend transport chunks.

## 5. Knowledge Base Layer

### 5.1 API And Service

- `knowledge_base_layer/api.py`: FastAPI app with health and search endpoints.
- `service.py`: policy search service, exact article handling, hybrid retrieval, parent context, and optional reranking.
- `bootstrap.py`: constructs Milvus store, embedding model, fusion, retriever, reranker, and service.
- `config.py`: Milvus, embedding, retrieval, PDF, and policy collection settings.

The search contract is policy-specific:

- request: query, top_k, optional law name, article id, as-of date, and region.
- response: list of knowledge hits.

### 5.2 Retrieval And Milvus

- `retrieval/milvus.py`: Milvus REST v2 storage and search implementation.
- `retrieval/retriever.py`: dense and BM25 recall with policy scalar filters.
- `retrieval/fusion.py`: reciprocal-rank fusion or weighted fusion.
- `retrieval/parent_context.py`: parent expansion for child chunks.
- `retrieval/reranker.py`: injectable reranker protocol.
- `retrieval/store.py`: vector store protocol.
- `embeddings.py`: local Sentence Transformer embedding loader.

The active policy alias setting is:

- `policy_collection_alias = "tender_qa_policy"`

### 5.3 PDF Ingestion And Publishing

- `data/pdf_sources.json`: policy PDF manifest.
- `data/pdf/`: source PDF snapshots.
- `ingestion/pdf.py`: PDF text extraction, article detection, parent-child chunking, and metadata generation.
- `ingestion/normalizer.py`: normalized fields, stable ids, and quality issue model.
- `ingestion/pipeline.py`: embedding, versioned collection writes, and alias publishing.
- `rebuild.py`: command-line rebuild entrypoint.

Only the local policy index is rebuildable at present.

## 6. Backend Layer

- `app_backend_layer/api.py`: FastAPI service lifecycle.
- `api_routes/chat.py`: loads session history, calls Agent stream, forwards chunks, and writes conversation history.
- `api_routes/sessions.py`: session listing, message reading, and deletion.
- `history/history_db.py`: SQLite storage for sessions and messages.
- `config.py`: Backend runtime settings, including Agent base URL.

Backend chat history is separate from Agent LangGraph workflow checkpoints.

## 7. Frontend Layer

- `app_frontend_layer/app.py`: Streamlit entrypoint.
- `api_client.py`: Backend API client.
- `components/chat_view.py`: streaming chat display.
- `components/sidebar.py`: session management and settings UI.
- `config.py`: Frontend runtime settings, including Backend URL.

## 8. Current Runtime Constraints

1. The project uses namespace-style packages; runtime commands must put the project root on `PYTHONPATH`.
2. Knowledge Base search requires an accessible Milvus service and a published `tender_qa_policy` alias.
3. Embedding dimensions must match the Milvus collection schema; the default dimension is 768.
4. Policy internet search requires an injected adapter. The built-in policy adapter is only a marker class.
5. SQL answers require an injected read-only executor and audit sink. The default app does not create a real database connection.
6. The two policy PDFs are snapshots through 2022 and should not be treated as a complete current law database.
7. There is no full automated test suite under `agent_layer/tests/` or `knowledge_base_layer/tests/`; top-level `test.py` is only a proxy diagnostic script.
