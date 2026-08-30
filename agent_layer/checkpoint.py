# coding: utf-8
# @Author: Wang Qingkang

from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class LangGraphCheckpointRuntime:
    def __init__(
        self,
        path: Path,
        exit_stack: AsyncExitStack,
        checkpointer: AsyncSqliteSaver,
    ) -> None:
        self.path = path
        self._exit_stack = exit_stack
        self.checkpointer = checkpointer

    @classmethod
    async def open(cls, path: Path) -> "LangGraphCheckpointRuntime":
        path.parent.mkdir(parents=True, exist_ok=True)
        exit_stack = AsyncExitStack()
        checkpointer = await exit_stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(path.as_posix())
        )
        return cls(path, exit_stack, checkpointer)

    def config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    async def close(self) -> None:
        await self._exit_stack.aclose()