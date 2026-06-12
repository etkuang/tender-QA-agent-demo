# coding: utf-8

import json
from pathlib import Path
from typing import Protocol

import aiosqlite


class CheckpointStore(Protocol):
    async def save(
        self,
        run_id: str,
        session_id: str,
        status: str,
        state: dict,
    ) -> None: ...

    async def load(self, run_id: str) -> dict | None: ...

    async def delete(self, run_id: str) -> None: ...


class SQLiteCheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    async def save(
        self,
        run_id: str,
        session_id: str,
        status: str,
        state: dict,
    ) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                INSERT INTO workflow_checkpoints(run_id, session_id, status, state_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (run_id, session_id, status, json.dumps(state, ensure_ascii=False)),
            )
            await connection.commit()

    async def load(self, run_id: str) -> dict | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                "SELECT session_id, status, state_json, updated_at FROM workflow_checkpoints WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "session_id": row[0],
            "status": row[1],
            "state": json.loads(row[2]),
            "updated_at": row[3],
        }

    async def delete(self, run_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("DELETE FROM workflow_checkpoints WHERE run_id = ?", (run_id,))
            await connection.commit()

    async def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_checkpoints(
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await connection.commit()
