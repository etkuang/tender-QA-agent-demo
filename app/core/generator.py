"""LLM生成器"""

from typing import List, Dict
import httpx

from app.config import settings


class LLMGenerator:
    """LLM生成器"""

    def __init__(self):
        self.api_key = settings.llm_api_key
        self.api_url = settings.llm_api_url
        self.model = settings.llm_model

    async def generate_with_history(
            self, query: str, context: List[Dict], collection: str, history: List[Dict] = None
    ) -> str:
        """生成答案（带历史）"""
        if not context:
            return "抱歉，没有找到相关信息。"

        prompt = self._build_prompt(query, context, collection, history)

        if self.api_key and self.api_url:
            return await self._call_llm(prompt)
        else:
            return self._fallback_answer(context, collection)

    async def _call_llm(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                return f"LLM API错误: {resp.status_code}"
        except Exception as e:
            return f"LLM调用失败: {str(e)}"

    def _system_prompt(self) -> str:
        return "你是招投标智能问答助手，请根据提供的参考信息准确回答问题。如果参考信息不足以回答问题，请如实告知，不要编造信息。"

    def _build_prompt(self, query: str, context: List[Dict], collection: str, history: List[Dict] = None) -> str:
        history_text = ""
        if history:
            history_parts = []
            for h in history[-3:]:
                history_parts.append(f"用户：{h['question']}\n助手：{h['answer']}")
            if history_parts:
                history_text = "【对话历史】\n" + "\n\n".join(history_parts) + "\n\n"

        context_text = "\n\n".join([
            f"【参考信息{i + 1}】\n{c.get('text', '')[:500]}"
            for i, c in enumerate(context[:3])
        ])

        return f"{history_text}【参考信息】\n{context_text}\n\n【当前问题】\n{query}\n\n【回答】"

    def _fallback_answer(self, context: List[Dict], collection: str) -> str:
        best = context[0].get("data", {})
        if collection == "bids":
            return f"找到相关招标项目：{best.get('project_name', '未知')}，中标人：{best.get('winner', '未知')}，中标金额：{best.get('winner_amount', '未知')}元"
        elif collection == "regulations":
            return f"找到相关法规内容：{best.get('text', '')[:300]}"
        else:
            return f"找到相关信息：{best}"
