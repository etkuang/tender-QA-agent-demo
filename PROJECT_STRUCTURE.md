# PROJECT_STRUCTURE.md

## 1. Project Purpose

This repository implements a four-process tender QA system:

- `knowledge_base_layer`: local policy knowledge-base service for PDF ingestion, embedding, Milvus retrieval, fusion, parent-context expansion, and index publishing.
- `agent_layer`: question decomposition, child-task graph execution, policy Self-RAG orchestration, data-domain workflows, SQL gateway contracts, website adapter contracts, model abstraction, and answer synthesis.
- `app_backend_layer`: backend API gateway and persistent chat-history storage.
- `app_frontend_layer`: Streamlit chat UI.

The runtime request path is:

Frontend -> Backend -> Agent -> Knowledge Base -> Milvus.

The Agent supports policy, tender, public-opinion, company, product, other, and unclear task categories. Multi-part questions are decomposed into child tasks, scheduled by dependency graph, executed through category-specific workflows, and synthesized into one final answer.

## 2. Design Principles

The current Agent design is built around these ideas:

1. Decompose before routing:
   - The Agent does not classify the whole user message into one route.
   - It decomposes the message into independently executable `ChildTask` objects.
   - Each child task carries its own category, dependency list, freshness need, and possible clarification question.

2. Execute partial work when possible:
   - `Category.UNCLEAR` tasks become unresolved child outcomes.
   - Independent solvable tasks still execute.
   - Tasks depending on unresolved or blocked prerequisites are marked blocked instead of aborting the whole request.

3. Keep workflow and child-task contracts separate:
   - `WorkflowResult` is the output of a workflow run and can carry workflow-level fields such as `run_id`.
   - `ChildTaskOutcome` is the output of executing a decomposed child task and carries `task_id`, status, answer, unresolved reason, evidence, citations, tool events, and model-call count.

4. Treat history as intent context, not evidence:
   - Recent conversation is passed to decomposition and workflow prompts only to resolve user intent, references, and omitted entities.
   - Final factual claims must come from evidence, structured data, or explicit workflow outputs.

5. Keep external capabilities injectable:
   - SQL execution, audit sinks, website adapters, policy internet search, and model providers are constructed through dependency boundaries.
   - Missing SQL or website adapters produce skipped or unresolved outcomes instead of fabricated data.

6. Preserve explainability through streaming:
   - Internal route, reasoning, progress, source, assistant, and error events are converted to Backend transport chunks.
   - `StreamEvent.content` is the authoritative user-facing text; the API layer only chunks and labels it for transport.

## 3. Top-Level Structure

- `main.py`: starts Knowledge Base, Agent, Backend, and Frontend services in order with dynamic local ports and environment injection.
- `knowledge_base_layer/`: policy document ingestion, Milvus retrieval, and `/search` API.
- `agent_layer/`: Agent API, composition root, question decomposition, child-task orchestration, model abstraction, workflows, retrieval adapters, SQL contracts, website adapters, and schemas.
- `app_backend_layer/`: session and message persistence plus Backend API routes.
- `app_frontend_layer/`: Streamlit frontend and Backend API client.
- `common/`: shared HTTP DTOs and logging utilities.
- `history/`: runtime location for Backend chat-history SQLite data.
- `pyproject.toml` and `uv.lock`: dependencies and lock file.
- `test.py`: standalone proxy-environment diagnostic script, not an automated test suite.

## 4. Service Startup And API Boundaries

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

## 5. Agent Layer

### 5.1 Composition And Lifecycle

- `agent_layer/api.py`: FastAPI entrypoint, NDJSON transport formatting, request cancellation checks, and application lifespan.
- `agent_layer/bootstrap.py`: async composition root. It creates model instances, retrieval clients, workflow objects, child-task workflow profiles, and the LangGraph checkpoint runtime.
- `agent_layer/app.py`: Agent application facade and child-task orchestration helpers.
- `agent_layer/config.py`: Agent runtime settings for LLM provider, Knowledge Base URL, streaming, history bounding, retrieval budgets, SQL limits, and LangGraph checkpoint path.
- `agent_layer/checkpoint.py`: `LangGraphCheckpointRuntime`, which owns `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` for the Agent process lifetime.

`build_application()` is asynchronous because it opens the LangGraph SQLite checkpointer before constructing the policy graph.

`TenderQAApplication.aclose()` closes the LangGraph checkpoint runtime during FastAPI shutdown.

### 5.2 Application Orchestration

`agent_layer/app.py` is split by responsibility:

- `ApplicationDependencies`: dependency container built by `bootstrap.py`.
- `TaskGraph`: validates child-task IDs, missing dependencies, and cycles; exposes ordered task IDs, task map, dependencies, dependents, and frontier tasks.
- `StreamEventFormatter`: owns user-facing route, execution-strategy, progress, source, and answer-chunk formatting.
- `ChildWorkflowRunner`: executes one child task by category and returns a `ChildTaskOutcome`.
- `ChildTaskExecutor`: schedules the child-task DAG, starts ready independent tasks, blocks dependent tasks when prerequisites fail, collects evidence/events, and calls final composite synthesis.
- `TenderQAApplication`: thin request-lifecycle facade. It handles quick responses, decomposition, task-graph creation, progress relay, source streaming, final answer streaming, and application shutdown cleanup.

The design keeps nested workflow details out of the API boundary. The API sees only `StreamEvent` values, while internal orchestration can reason in richer objects such as `TaskGraph`, `WorkflowResult`, and `ChildTaskOutcome`.

### 5.3 Question Decomposition

- `question_decomposition/prompts.py`: prompt for decomposition-first question understanding and anaphora resolution in each child task.
- `question_decomposition/chain.py`: `QuestionDecomposer`, which uses structured output to produce a JSON-array-shaped `ChildTaskList` and returns `list[ChildTask]`.

The decomposer creates self-contained child-task questions. It no longer produces a separate standalone question, confidence score, global route decision, or classification-level entity list.

Important schema rules:

- `ChildTaskList` must contain at least one child task.
- `Category.UNCLEAR` child tasks must provide `clarification_question`.
- Executable child tasks must not provide `clarification_question`.
- `depends_on` may only point to generated child-task IDs and is later validated by `TaskGraph`.

There is no active `WorkflowRouter` or `RoutePlan` path. Routing happens when each child task is executed by `ChildWorkflowRunner`.

### 5.4 Quick Responses

- `conversation/quick_classifier.py`: structured classifier for pure conversational shortcuts.
- `conversation/quick_responses.py`: canned responses and route text for greetings, thanks, goodbye, capabilities, wellbeing, acknowledgement, compliment, apology, and empty input.
- `conversation/fallback_messages.py`: shared no-evidence and source-unavailable fallback text.

Quick-response classification is intentionally non-fatal. If it fails, it returns `QuickResponseType.NONE`, and the Agent continues into normal decomposition and workflow execution.

### 5.5 Model Abstraction And Concurrency

- `models.py`: `ModelFactory` builds either OpenAI-compatible chat models or local Hugging Face chat models behind the `BaseChatModel` contract.
- `local_llm.py`: local Hugging Face runtime and chat model wrapper. It supports tokenizer chat templates, async generation through `asyncio.to_thread`, timeout handling, and JSON extraction for structured output.
- `model_concurrency.py`: shared `ModelConcurrencyGate` plus `ConcurrencyLimitedChatModel`.

Design points:

- Structured and answer models share the same model-provider abstraction.
- Local Hugging Face mode reuses one loaded runtime through `ModelFactory`.
- Async model calls are gated to prevent uncontrolled concurrency, especially for local GPU-backed generation.
- Structured outputs use `with_structured_output(..., method="json_mode")` and are validated by Pydantic schemas.

### 5.6 Policy Workflow

Policy execution is split between:

- `workflows/policy.py`: policy prompts and LangChain components:
  - `PolicyQueryParser`
  - `PolicyAssessmentChain`
  - policy answer prompt
- `workflows/policy_graph.py`: LangGraph `PolicyWorkflow`.

`PolicyWorkflow` builds a `StateGraph` with these nodes:

1. `parse_policy`: extract law name, article id, date, and region.
2. `retrieve`: retrieve local policy evidence through the Knowledge Base.
3. `assess`: grade evidence sufficiency with structured output.
4. `policy_internet`: optional official-internet supplement if an injected adapter is configured.
5. `synthesize`: produce the policy answer with citations.

Conditional edges route from assessment back to retrieval, to official internet search, or to synthesis. If official internet evidence is added, the graph reassesses the expanded evidence set.

The graph is compiled with the LangGraph SQLite checkpointer. Each policy run uses its `run_id` as the LangGraph `thread_id`, so checkpoints are stored by run. This is why `WorkflowResult.run_id` remains workflow-level state rather than child-task identity.

### 5.7 Self-RAG Helpers

- `workflows/self_rag.py`: evidence merging, follow-up query selection, and final evidence selection.
- `workflows/common.py`: child-task workflow profile model, evidence formatting, citation validation, citation building, source-section enforcement, and evidence ranking.

Self-RAG retrieval budgets are configured by:

- `retrieval_batch_size`
- `max_retrieval_rounds`
- `max_follow_up_queries`
- `evidence_chunk_chars`
- `evidence_context_chars`

### 5.8 Data-Domain Workflows

`workflows/child_task.py` owns the shared `GeneralChildTaskWorkflow` used for tender, public-opinion, company, product, and other child tasks. Category profiles are constructed in `bootstrap.py` and declare a description, tool pool, and ordered tool-preference tiers.

Supporting data-domain implementation lives under `data_domain/`:

- `chains.py` and `prompts.py`: intent, SQL generation and repair, and answer chains plus prompt-input formatting.
- `schemas.py`: intent, semantic-context, SQL request, validation, audit, and analysis models.
- `context/`: category-aware tables, relationships, metrics, glossary terms, examples, and context retrieval.
- `sql/`: read-only gateway and SQLGlot policy validation.
- `analysis.py`: deterministic SQL-result analysis and evidence construction.
- `web.py`: category-keyed website supplementation.

The active general child-task flow:

1. Classifies the child-task intent.
2. Iterates the profile's configured tool tiers.
3. For SQL, retrieves semantic context, generates and validates a read-only query, performs bounded repair, executes through the injected gateway, analyzes the result, and builds evidence.
4. For website retrieval, invokes the configured category client and converts results into evidence.
5. Stops after a tier yields eligible evidence, or uses the configured model-only fallback.
6. Synthesizes a citation-checked child-task answer.

SQL and website capabilities are dependency-injected. Missing adapters are reported as skipped. Source failures can be converted into unresolved workflow results so final synthesis can explain partial completion.

Data-domain workflows currently do not use checkpoint persistence. They can be migrated independently if resumable data-domain execution is required later.

### 5.9 SQL Boundary

- `data_domain/sql/gateway.py`: read-only executor, audit-sink, and gateway contracts plus the audited gateway implementation.
- `data_domain/sql/validator.py`: uses `sqlglot` to parse a single read-only SELECT/CTE/UNION statement, restrict tables and columns, block sensitive columns and dangerous functions, and enforce LIMIT.
- `data_domain/context/catalog.py`: default semantic table, relationship, metric, glossary, and approved-example catalog.
- `data_domain/schemas.py`: SQL candidate, execution-request, validation-result, audit-event, and data-context models.

The current SQL dialect is configured through Agent settings:

- `sql_dialect = "sqlite"`

The only remaining SQL runtime row cap is:

- `sql_max_rows`: maximum rows returned by a read-only SQL query and enforced as LIMIT.

The default Agent application still does not construct a real SQL executor. A read-only executor and audit sink must be injected for SQL-backed data-domain answers.

### 5.10 Website Adapters

- `adapters/base.py`: shared website adapter protocol, request model, config model, and base adapter behavior.
- `adapters/policy_internet.py`, `tender_web.py`, `public_opinion_web.py`, `company_web.py`, and `product_web.py`: category marker adapters.

The adapter subclasses currently declare only their expected category. Real request construction and response parsing must be implemented by deployment-specific adapters.

For policy internet evidence, the workflow currently trusts `SourceTier.OFFICIAL` from the injected adapter. It no longer performs domain-suffix verification.

### 5.11 Schemas And Events

- `schemas.py`: categories, child tasks, quick-response decisions, policy queries, evidence, citations, data results, website queries, tool events, workflow results, child-task outcomes, and stream events.
- `StreamEventType`: route, progress, reasoning summary, source, assistant delta, and error.
- `agent_layer/api.py`: adapts internal stream events to Backend transport chunks.

Important contract separation:

- `WorkflowResult`: workflow-level answer, evidence, citations, tool events, model-call count, run ID, status, and unresolved reason.
- `ChildTaskOutcome`: task-level status and output used for dependency tracking and final synthesis.
- `ToolEvent`: workflow progress record consumed by `StreamEventFormatter`.

## 6. Knowledge Base Layer

### 6.1 API And Service

- `knowledge_base_layer/api.py`: FastAPI app with health and search endpoints.
- `service.py`: policy search service, exact article handling, hybrid retrieval, parent context, and optional reranking.
- `bootstrap.py`: constructs Milvus store, embedding model, fusion, retriever, reranker, and service.
- `config.py`: Milvus, embedding, retrieval, PDF, and policy collection settings.

The search contract is policy-specific:

- request: query, top_k, optional law name, article id, as-of date, and region.
- response: list of knowledge hits.

### 6.2 Retrieval And Milvus

- `retrieval/milvus.py`: Milvus REST v2 storage and search implementation.
- `retrieval/retriever.py`: dense and BM25 recall with policy scalar filters.
- `retrieval/fusion.py`: reciprocal-rank fusion or weighted fusion.
- `retrieval/parent_context.py`: parent expansion for child chunks.
- `retrieval/reranker.py`: injectable reranker protocol.
- `retrieval/store.py`: vector store protocol.
- `embeddings.py`: local Sentence Transformer embedding loader.

The active policy alias setting is:

- `policy_collection_alias = "tender_qa_policy"`

### 6.3 PDF Ingestion And Publishing

- `data/pdf_sources.json`: policy PDF manifest.
- `data/pdf/`: source PDF snapshots.
- `ingestion/pdf.py`: PDF text extraction, article detection, parent-child chunking, and metadata generation.
- `ingestion/normalizer.py`: normalized fields, stable IDs, and quality issue model.
- `ingestion/pipeline.py`: embedding, versioned collection writes, and alias publishing.
- `rebuild.py`: command-line rebuild entrypoint.

Only the local policy index is rebuildable at present.

## 7. Backend Layer

- `app_backend_layer/api.py`: FastAPI service lifecycle and request-ID middleware.
- `api_routes/chat.py`: loads session history, calls Agent stream, forwards chunks, and writes conversation history.
- `api_routes/sessions.py`: session listing, message reading, and deletion.
- `history/history_db.py`: SQLite storage for sessions and messages.
- `config.py`: Backend runtime settings, including Agent base URL.

Backend chat history is separate from Agent LangGraph workflow checkpoints.

## 8. Frontend Layer

- `app_frontend_layer/app.py`: Streamlit entrypoint.
- `api_client.py`: Backend API client with request context and streaming support.
- `components/chat_view.py`: streaming chat display.
- `components/sidebar.py`: session management and settings UI.
- `config.py`: Frontend runtime settings, including Backend URL.

## 9. Technical Design Points Worth Preserving

1. The Agent facade is intentionally thin:
   - `TenderQAApplication` coordinates the request lifecycle but delegates formatting, child workflow execution, and child-task graph scheduling.

2. Child-task execution is a DAG scheduler:
   - Independent tasks can run concurrently.
   - Dependents start as soon as their own prerequisites are solved.
   - Failed or unresolved prerequisites block only downstream dependents.

3. The system favors explicit contracts over implicit state:
   - Pydantic models define structured model outputs, stream events, tool events, workflow outputs, evidence, and transport DTOs.

4. The project separates evidence from conversation context:
   - History helps understand the user's question.
   - Evidence and data sources ground factual claims.

5. Source failures are recoverable at the child-task level:
   - SQL and website failures can become unresolved child outcomes.
   - Final synthesis can still report completed work and explain missing parts.

6. The policy workflow is resumable separately from chat history:
   - LangGraph checkpoints are keyed by policy workflow `run_id`.
   - Backend session history remains a separate persistence concern.

7. Model provider details are isolated:
   - OpenAI-compatible and local Hugging Face models are both exposed as `BaseChatModel`.
   - Concurrency gating prevents local model overload and bounds API parallelism.

8. Transport formatting is intentionally simple:
   - The application layer owns semantic wording.
   - The API layer only maps stream event types to assistant, reasoning, or error chunks.

## 10. Current Runtime Constraints

1. The project uses namespace-style packages; runtime commands must put the project root on `PYTHONPATH`.
2. Knowledge Base search requires an accessible Milvus service and a published `tender_qa_policy` alias.
3. Embedding dimensions must match the Milvus collection schema; the default dimension is 768.
4. Policy internet search requires an injected adapter. The built-in policy adapter is only a marker class.
5. SQL answers require an injected read-only executor and audit sink. The default app does not create a real database connection.
6. Website-backed data-domain answers require deployment-specific website adapters.
7. Local Hugging Face mode requires optional runtime dependencies and enough local hardware for the configured model.
8. The two policy PDFs are snapshots through 2022 and should not be treated as a complete current law database.
9. There is no full automated test suite under `agent_layer/tests/` or `knowledge_base_layer/tests/`; top-level `test.py` is only a proxy diagnostic script.
