# coding: utf-8

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent_layer.config import Settings
from agent_layer.local_llm import LocalHuggingFaceChatModel, LocalHuggingFaceRuntime


class ModelFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._local_runtime = None

    def build_chat_model(self, temperature: float = 0.0, max_tokens: int | None = None) -> BaseChatModel:
        resolved_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_tokens
        if self.settings.llm_provider == "local_huggingface":
            return LocalHuggingFaceChatModel(
                runtime=self._get_local_runtime(),
                temperature=temperature,
                max_tokens=resolved_max_tokens,
                timeout=self.settings.llm_timeout_seconds,
            )
        if self.settings.llm_api_key is None or self.settings.llm_api_url is None:
            raise ValueError("llm_api_key and llm_api_url are required for openai_compatible mode.")
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_url,
            temperature=temperature,
            max_tokens=resolved_max_tokens,
            timeout=self.settings.llm_timeout_seconds,
        )

    def _get_local_runtime(self) -> LocalHuggingFaceRuntime:
        if self._local_runtime is None:
            self._local_runtime = LocalHuggingFaceRuntime(
                model_name=self.settings.llm_model,
                device_map=self.settings.llm_local_device_map,
                torch_dtype=self.settings.llm_local_torch_dtype,
                trust_remote_code=self.settings.llm_local_trust_remote_code,
            )
        return self._local_runtime
