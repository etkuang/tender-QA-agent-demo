from typing import List, Dict, Optional
import httpx
from config import settings


class LLMGenerator:
    def __init__(self):
        self.api_key = settings.llm_api_key
        self.api_url = settings.llm_api_url
        self.model = settings.llm_model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        """生成回答，支持自定义 temperature"""
        if not self.api_key or not self.api_url:
            return "请配置 LLM API Key 和 URL"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._call_llm(messages, temperature)

    async def chat(self, messages: List[Dict], temperature: float = 0.3) -> str:
        """对话接口，支持自定义 temperature"""
        return await self._call_llm(messages, temperature)

    async def _call_llm(self, messages: List[Dict], temperature: float = 0.3) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": settings.llm_max_tokens
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                return f"API错误: {resp.status_code}"
        except Exception as e:
            return f"调用失败: {str(e)}"


    async def quick_generate(self, prompt: str, max_tokens: int = 200) -> str:
        """快速生成，用于简单任务（如问题重写）"""
        if not self.api_key or not self.api_url:
            return prompt

        messages = [{"role": "user", "content": prompt}]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_tokens
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                return prompt
        except Exception:
            return prompt

