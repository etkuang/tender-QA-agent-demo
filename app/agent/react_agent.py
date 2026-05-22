"""ReAct Agent - Thought → Action → Observation 循环（带会话记忆）"""
import json
import re
from typing import List, Dict, Any, Optional
from app.core.retriever import HybridRetriever
from app.core.generator import LLMGenerator
from app.agent.agent_tools import TOOL_CLASSES, BaseTool
from config import settings

# ReAct Prompt 模板（增强版，包含历史对话）
REACT_PROMPT = """你是一个招投标法规专家助手。你可以使用工具来检索信息，然后基于检索结果回答问题。

## 历史对话记录（最近 5 轮）
{history}

## 当前用户问题
{question}

## 可用工具
{tools_description}

## 工具调用格式
**重要：Action Input 必须是 JSON 格式**

示例：
Thought: 需要查询围标的定义
Action: search_law
Action Input: {{"query": "串通投标的定义"}}

或者查询法条：
Thought: 需要查第XX条
Action: get_article
Action Input: {{"article_num": "XX"}}

## 重要规则
1. 每轮只能调用一个工具
2. Action Input 必须使用双花括号 {{}} 包裹的 JSON 格式
3. search_law 工具的参数是 query
4. get_article 工具的参数是 article_num（可选 law_name）
5. 信息足够时输出 Final Answer
6. 【重要】如果用户使用了指代词（如"上面提到的"、"这个处罚是什么"），请结合历史对话理解问题
7. 【重要】对于比较类问题（如"A和B的区别"），请先分别检索A和B的相关信息，再进行比较。不要只检索其中一个。
8. 尽量在一次响应中规划多个检索步骤，避免浪费步数。
请继续输出（必须包含 Thought 和 Action，或 Final Answer）：
"""


class ReActAgent:
    """ReAct Agent - 思考→行动→观察 循环（支持会话记忆）"""

    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator, max_steps: int = 5, session_manager=None):
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.session_manager = session_manager
        self.tools: Dict[str, BaseTool] = {}

        for tool_name, tool_class in TOOL_CLASSES.items():
            self.tools[tool_name] = tool_class(retriever)

    def _get_tools_description(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _get_history_context(self, session_id: str, max_messages: int = 10) -> str:
        """获取格式化的历史对话上下文（默认10条=5轮）"""
        if not self.session_manager or not session_id:
            return "（无历史记录）"

        try:
            history = self.session_manager.get_chat_history(session_id)
            messages = history.messages[-max_messages:] if max_messages > 0 else history.messages

            if not messages:
                return "（无历史记录）"

            context_lines = ["以下是最近的对话历史："]
            for msg in messages:
                role = "用户" if msg.type == "human" else "助手"
                content = msg.content[:800]  # Agent 需要更长上下文
                context_lines.append(f"[{role}]: {content}")

            return "\n".join(context_lines)
        except Exception as e:
            print(f"   ⚠️ 获取历史上下文失败: {e}")
            return "（无历史记录）"

    def _parse_action(self, text: str) -> Optional[Dict]:
        """解析 Action，增强容错能力"""
        # 1. 提取 Action
        action_match = re.search(r'Action:\s*(\w+)', text)
        if not action_match:
            return None

        action = action_match.group(1)

        # 2. 尝试多种方式提取 Action Input

        # 方式1：标准 JSON 格式（单行或多行）
        json_match = re.search(r'Action Input:\s*(\{[\s\S]+?\})', text)
        if json_match:
            try:
                action_input = json.loads(json_match.group(1))
                return {"action": action, "action_input": action_input}
            except json.JSONDecodeError:
                pass

        # 方式2：双引号包裹的纯字符串
        quoted_match = re.search(r'Action Input:\s*"([^"]+)"', text)
        if quoted_match:
            query = quoted_match.group(1).strip()
            if action == "search_law":
                return {"action": action, "action_input": {"query": query}}
            elif action == "get_article":
                if query.isdigit():
                    return {"action": action, "action_input": {"article_num": query}}
                else:
                    return {"action": action, "action_input": {"query": query}}

        # 方式3：单引号包裹
        quoted_match = re.search(r"Action Input:\s*'([^']+)'", text)
        if quoted_match:
            query = quoted_match.group(1).strip()
            if action == "search_law":
                return {"action": action, "action_input": {"query": query}}
            elif action == "get_article":
                if query.isdigit():
                    return {"action": action, "action_input": {"article_num": query}}
                else:
                    return {"action": action, "action_input": {"query": query}}

        # 方式4：普通文本（到行尾）
        plain_match = re.search(r'Action Input:\s*(.+?)(?=\n\s*(?:Thought|Action|Final Answer)|\n*$)', text, re.DOTALL)
        if plain_match:
            query = plain_match.group(1).strip().strip('"').strip("'")
            if query and query != "{}" and query != "":
                if action == "search_law":
                    return {"action": action, "action_input": {"query": query}}
                elif action == "get_article":
                    if query.isdigit():
                        return {"action": action, "action_input": {"article_num": query}}
                    else:
                        return {"action": action, "action_input": {"query": query}}

        # 方式5：从文本中提取 query 参数（JSON 格式但解析失败时的降级）
        query_match = re.search(r'"query"\s*:\s*"([^"]+)"', text)
        if query_match and action == "search_law":
            return {"action": action, "action_input": {"query": query_match.group(1)}}

        article_match = re.search(r'"article_num"\s*:\s*"(\d+)"', text)
        if article_match and action == "get_article":
            return {"action": action, "action_input": {"article_num": article_match.group(1)}}

        # 方式6：直接的数字（可能是 article_num）
        if action == "get_article":
            num_match = re.search(r'Action Input:\s*(\d+)', text)
            if num_match:
                return {"action": action, "action_input": {"article_num": num_match.group(1)}}

        # 最后降级：使用原问题作为 query
        return {"action": action, "action_input": {}}

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """提取 Final Answer，支持多种格式"""
        # 方式1：标准格式
        match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 方式2：LLM 直接输出了答案（没有 Final Answer 标记）
        # 如果响应很长且没有 Action，说明可能是答案
        if not re.search(r'Action:', text):
            if len(text) > 100:
                return text.strip()

        return None

    def _extract_thought(self, text: str) -> str:
        match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "思考中"

    async def run(self, question: str, session_id: str = "") -> str:
        """运行 ReAct Agent，支持会话记忆（5 轮历史）"""
        print(f"\n🧠 [ReActAgent] 处理问题: {question}")
        print(f"   session_id={session_id or 'none'}")
        print(f"   max_steps={self.max_steps}")

        # 获取历史对话上下文（5 轮 = 10 条消息）
        base_history = self._get_history_context(session_id, max_messages=10)

        MAX_HISTORY_STEPS = 3  # 只保留最近3轮
        step_records = []  # 存储每一步的文本

        for step in range(self.max_steps):
            print(f"\n   Step {step + 1}: 思考中...")

            # 构建 history_context：基础历史 + 最近几步的检索过程
            history_context = base_history
            if step_records:
                recent_steps = step_records[-MAX_HISTORY_STEPS:]
                history_context += "\n\n## 刚才的检索过程\n" + "\n\n".join(recent_steps)

            prompt = REACT_PROMPT.format(
                history=history_context,
                question=question,
                tools_description=self._get_tools_description(),
            )

            try:
                response = await self.llm.generate(prompt, temperature=settings.react_temperature)
                print(f"   📝 LLM 响应:\n{response[:500]}...")

                final_answer = self._extract_final_answer(response)
                if final_answer:
                    print(f"   ✅ 得到最终答案")
                    return final_answer

                if not re.search(r'Action:', response):
                    if len(response) > 100:
                        print(f"   ✅ 检测到直接答案（无 Action 标记）")
                        return response.strip()

                action_info = self._parse_action(response)
                if not action_info:
                    print(f"   ⚠️ 无法解析 Action，使用默认检索")
                    action_info = {"action": "search_law", "action_input": {"query": question}}

                action = action_info["action"]
                action_input = action_info.get("action_input", {})

                if action not in self.tools:
                    print(f"   ❌ 未知工具: {action}")
                    observation = f"错误：工具 '{action}' 不存在。可用工具: {list(self.tools.keys())}"
                else:
                    print(f"   🔧 执行工具: {action}({action_input})")
                    tool = self.tools[action]
                    observation = await tool.run(**action_input)
                    print(f"   👁️ Observation: {observation[:200]}...")

                thought = self._extract_thought(response)

                # 存储这一步的记录（截断过长内容）
                step_text = f"Thought: {thought[:200]}\nAction: {action}\nObservation: {observation[:500]}"
                step_records.append(step_text)

            except Exception as e:
                print(f"   ❌ ReAct Agent 错误: {e}")
                from app.tools.regulation_tools import summarize
                results = self.retriever.search(question, "regulations", top_k=5)
                if results:
                    return await summarize(results, question, self.llm)
                return f"处理失败: {str(e)}"

        # 达到最大步数，强制总结
        print(f"   ⏰ 达到最大步数 {self.max_steps}，强制总结")

        # 从 step_records 中提取所有 observation
        all_observations = []
        for record in step_records:
            # 提取 Observation 部分

            obs_match = re.search(r'Observation: (.+?)(?=Thought:|$)', record, re.DOTALL)
            if obs_match:
                all_observations.append(obs_match.group(1).strip())

        context = "\n\n".join(all_observations[:5])

        if not context:
            results = self.retriever.search(question, "regulations", top_k=5)
            from app.tools.regulation_tools import summarize
            return await summarize(results, question, self.llm)

        force_prompt = f"""基于以下检索到的信息回答问题。

用户问题：{question}

检索到的相关信息（这是你通过工具获取到的唯一信息来源）：
{context[:3000]}

【严格规则】：
1. 如果上述检索结果为空、不相关，或者不足以回答问题，请直接回复："抱歉，根据现有知识库未找到相关信息。"
2. 不要编造任何检索结果中没有的内容。
3. 只回答有检索结果依据的部分。

请严格遵守以上规则。"""

        try:
            final_answer = await self.llm.generate(force_prompt, temperature=settings.react_temperature)
            return final_answer
        except Exception as e:
            return f"经过 {self.max_steps} 步搜索，未能得到完整答案。请尝试更具体的问题。"