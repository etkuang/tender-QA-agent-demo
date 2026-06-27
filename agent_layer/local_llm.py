# coding: utf-8
# @Author: Wang Qingkang

import asyncio
import json
from json import JSONDecodeError
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr


class LocalHuggingFaceRuntime:
    def __init__(
        self,
        model_name: str,
        device_map: str,
        torch_dtype: str,
        trust_remote_code: bool,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=self._resolve_torch_dtype(torch, torch_dtype),
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()

    @staticmethod
    def _resolve_torch_dtype(torch: Any, torch_dtype: str) -> Any:
        if torch_dtype == "auto":
            return "auto"
        return getattr(torch, torch_dtype)

    def generate(
        self,
        messages: list[BaseMessage],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        input_ids = self._tokenizer.apply_chat_template(
            [self._message_to_dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with self._torch.inference_mode():
            outputs = self._model.generate(**generation_kwargs)
        text = self._tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        return self._truncate_at_stop(text, stop)

    @staticmethod
    def _message_to_dict(message: BaseMessage) -> dict:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        return {"role": role, "content": message.content}

    @staticmethod
    def _truncate_at_stop(text: str, stop: list[str] | None) -> str:
        if not stop:
            return text
        indexes = [text.find(item) for item in stop if item in text]
        if not indexes:
            return text
        return text[: min(indexes)]


class LocalHuggingFaceChatModel(BaseChatModel):
    model_name: str
    temperature: float = 0.0
    max_tokens: int
    timeout: float

    _runtime: LocalHuggingFaceRuntime = PrivateAttr()

    def __init__(
        self,
        runtime: LocalHuggingFaceRuntime,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ):
        super().__init__(
            model_name=runtime.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._runtime = runtime

    @property
    def _llm_type(self) -> str:
        return "local_huggingface"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._runtime.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.wait_for(
            asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs),
            timeout=self.timeout,
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        if method != "json_mode":
            raise NotImplementedError("Local Hugging Face structured output only supports json_mode.")
        if include_raw:
            raise NotImplementedError("Local Hugging Face structured output does not support include_raw.")
        return RunnableLambda(
            lambda prompt_value: self._invoke_structured(prompt_value, schema),
            afunc=lambda prompt_value: self._ainvoke_structured(prompt_value, schema),
        )

    def _invoke_structured(self, prompt_value: Any, schema: type[BaseModel]) -> BaseModel:
        response = self.invoke(prompt_value.to_messages())
        return schema.model_validate_json(self._extract_json_text(response.content))

    async def _ainvoke_structured(self, prompt_value: Any, schema: type[BaseModel]) -> BaseModel:
        response = await self.ainvoke(prompt_value.to_messages())
        return schema.model_validate_json(self._extract_json_text(response.content))

    def _extract_json_text(self, content: str) -> str:
        text = self._strip_markdown_fence(content.strip())
        try:
            json.loads(text)
            return text
        except JSONDecodeError:
            start = self._first_json_start(text)
            end = self._matching_json_end(text, start)
            return text[start:end + 1]

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _first_json_start(text: str) -> int:
        object_start = text.find("{")
        array_start = text.find("[")
        starts = [index for index in [object_start, array_start] if index >= 0]
        if not starts:
            raise ValueError("Local model did not return a JSON object or array.")
        return min(starts)

    @staticmethod
    def _matching_json_end(text: str, start: int) -> int:
        stack = []
        in_string = False
        escaped = False
        pairs = {"{": "}", "[": "]"}
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in pairs:
                stack.append(pairs[char])
                continue
            if stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return index
        raise ValueError("Local model returned incomplete JSON.")