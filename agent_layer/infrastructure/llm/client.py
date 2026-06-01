from typing import Dict, List, Optional

import httpx

from agent_layer.infrastructure.config.agent_settings import settings


class LLMRequestError(Exception):
    pass


class LLMGenerator:
    def __init__(self):
        self.api_key = settings.llm_api_key
        self.api_url = settings.llm_api_url
        self.model = settings.llm_model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        if not self.api_key or not self.api_url:
            raise LLMRequestError("Missing llm_api_key or llm_api_url")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._call_llm(messages, temperature, settings.llm_max_tokens)

    async def chat(self, messages: List[Dict], temperature: float = 0.3) -> str:
        if not self.api_key or not self.api_url:
            raise LLMRequestError("Missing llm_api_key or llm_api_url")
        return await self._call_llm(messages, temperature, settings.llm_max_tokens)

    async def quick_generate(self, prompt: str, max_tokens: int = 200) -> str:
        if not self.api_key or not self.api_url:
            return prompt

        messages = [{"role": "user", "content": prompt}]
        try:
            return await self._call_llm(messages, 0.0, max_tokens)
        except Exception:
            return prompt

    async def _call_llm(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )

        if response.status_code != 200:
            raise LLMRequestError(f"LLM API status={response.status_code} body={response.text[:400]}")

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise LLMRequestError("LLM response content is not string")
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Invalid LLM payload: {payload}") from exc
