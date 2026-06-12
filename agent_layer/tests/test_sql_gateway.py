# coding: utf-8

import unittest
from types import SimpleNamespace

from agent_layer.errors import SQLValidationError
from agent_layer.schemas import DataResult, ResearchTask, SessionContext
from agent_layer.sql.gateway import ReadOnlySQLGateway
from agent_layer.sql.schemas import SQLCandidate, SQLValidationResult, ViewSchema


class FakeValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, statement: str, schemas: list[ViewSchema]) -> SQLValidationResult:
        self.calls += 1
        if self.calls == 1:
            raise SQLValidationError
        return SQLValidationResult(
            statement="SELECT company_name FROM company_profile_view LIMIT 20",
            tables=["company_profile_view"],
            columns=["company_name"],
            limit=20,
        )


class FakeExecutor:
    async def execute_read_only(
        self,
        statement: str,
        parameters: dict,
        timeout_seconds: float,
        max_rows: int,
        runtime_context: SessionContext,
    ) -> DataResult:
        return DataResult(
            columns=["company_name"],
            rows=[{"company_name": "示例企业"}],
            row_count=1,
            query_id="query-1",
        )


class FakeAuditSink:
    def __init__(self):
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


class GatewayUnderTest(ReadOnlySQLGateway):
    def __init__(self, validator, executor, schemas, settings, audit_sink, candidates):
        self.validator = validator
        self.executor = executor
        self.schemas = schemas
        self.settings = settings
        self.audit_sink = audit_sink
        self.candidates = candidates

    async def _generate(self, task, schemas, feedback):
        return self.candidates.pop(0)


class SQLGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_retry_and_success_are_audited(self):
        validator = FakeValidator()
        audit_sink = FakeAuditSink()
        view = ViewSchema(
            name="company_profile_view",
            description="企业公开画像",
            columns={"company_name": "企业名称"},
        )
        candidates = [
            SQLCandidate(statement="DELETE FROM company_profile_view"),
            SQLCandidate(statement="SELECT company_name FROM company_profile_view LIMIT 20"),
        ]
        gateway = GatewayUnderTest(
            validator,
            FakeExecutor(),
            {view.name: view},
            SimpleNamespace(sql_statement_timeout_seconds=5.0, sql_max_rows=20),
            audit_sink,
            candidates,
        )

        result = await gateway.execute(
            ResearchTask(task_id="company-query", goal="查询企业", preferred_source="sql"),
            [view.name],
            SessionContext(session_id="session-1"),
        )

        self.assertEqual(result.query_id, "query-1")
        self.assertEqual([event.status for event in audit_sink.events], ["validation_failed", "completed"])
        self.assertEqual(audit_sink.events[-1].row_count, 1)


if __name__ == "__main__":
    unittest.main()
