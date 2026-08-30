# coding: utf-8
# @Author: Wang Qingkang

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr


class ModelConcurrencyGate:
    def __init__(self, limit: int | None):
        self._semaphore = None
        if limit is not None:
            self._semaphore = asyncio.Semaphore(limit)

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if self._semaphore is None:
            return await operation()
        async with self._semaphore:
            return await operation()


class ConcurrencyLimitedChatModel(BaseChatModel):
    _model = PrivateAttr()
    _gate = PrivateAttr()

    def __init__(self, model: BaseChatModel, gate: ModelConcurrencyGate):
        super().__init__()
        self._model = model
        self._gate = gate

    @property
    def _llm_type(self) -> str:
        return f"concurrency_limited_{self._model._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return self._model._identifying_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._model._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self._gate.run(
            lambda: self._model._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        runnable = self._model.with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            **kwargs,
        )
        return RunnableLambda(
            lambda input_value: runnable.invoke(input_value),
            afunc=lambda input_value: self._gate.run(lambda: runnable.ainvoke(input_value)),
        )