# coding: utf-8

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class LangGraphCheckpointRuntime:
    """Owns the LangGraph SQLite checkpointer for the Agent process lifetime."""

    def __init__(self, path: Path, context: Any, saver: AsyncSqliteSaver):
        self.path = path
        self.context = context
        self.saver = saver

    @classmethod
    async def open(cls, path: Path) -> "LangGraphCheckpointRuntime":
        path.parent.mkdir(parents=True, exist_ok=True)
        context = AsyncSqliteSaver.from_conn_string(path.as_posix())
        saver = await context.__aenter__()
        return cls(path, context, saver)

    def config(self, run_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": run_id}}

    async def load(self, run_id: str) -> CheckpointTuple | None:
        return await self.saver.aget_tuple(self.config(run_id))

    async def history(self, run_id: str) -> list[CheckpointTuple]:
        output = []
        async for checkpoint in self.saver.alist(self.config(run_id)):
            output.append(checkpoint)
        return output

    async def close(self) -> None:
        await self.context.__aexit__(None, None, None)
