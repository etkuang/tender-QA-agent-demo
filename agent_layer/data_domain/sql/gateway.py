# coding: utf-8
# @Author: Wang Qingkang

import time
from typing import Protocol

from common.logger import get_logger
from agent_layer.errors import SQLExecutionError
from agent_layer.schemas import DataResult
from agent_layer.data_domain.schemas import DataAuditEvent, SQLExecutionRequest

logger = get_logger("agent.data_domain.sql.gateway")


class SQLExecutor(Protocol):
    async def execute_read_only(
        self,
        statement: str,
        parameters: dict,
        timeout_seconds: float,
        max_rows: int,
    ) -> DataResult: ...


class SQLAuditSink(Protocol):
    async def record(self, event: DataAuditEvent) -> None: ...


class SQLGateway(Protocol):
    async def execute(self, request: SQLExecutionRequest) -> DataResult: ...


class NoopSQLAuditSink:
    async def record(self, event: DataAuditEvent) -> None:
        return None


class ReadOnlySQLGateway:
    def __init__(
        self,
        executor: SQLExecutor,
        audit_sink: SQLAuditSink | None = None,
    ):
        self.executor = executor
        self.audit_sink = audit_sink or NoopSQLAuditSink()

    async def execute(self, request: SQLExecutionRequest) -> DataResult:
        started = time.perf_counter()
        try:
            result = await self.executor.execute_read_only(
                request.statement,
                request.parameters,
                request.timeout_seconds,
                request.max_rows,
            )
        except Exception as exc:
            await self.audit_sink.record(
                DataAuditEvent(
                    request_id=request.request_id,
                    domain=request.domain,
                    statement=request.statement,
                    parameter_names=sorted(request.parameters),
                    status="execution_failed",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_type=exc.__class__.__name__,
                )
            )
            logger.warning("read-only SQL execution failed | request_id=%s", request.request_id, exc_info=True)
            raise SQLExecutionError from exc
        await self.audit_sink.record(
            DataAuditEvent(
                request_id=request.request_id,
                domain=request.domain,
                statement=request.statement,
                parameter_names=sorted(request.parameters),
                status="completed",
                duration_ms=(time.perf_counter() - started) * 1000,
                row_count=result.row_count,
                query_id=result.query_id,
            )
        )
        return result