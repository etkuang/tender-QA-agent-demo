# coding: utf-8

from langchain_openai import ChatOpenAI

from agent_layer.config import Settings


class ModelFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_chat_model(self, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_url,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
            timeout=self.settings.llm_timeout_seconds,
        )
