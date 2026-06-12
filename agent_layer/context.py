# coding: utf-8

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from common.logger import get_logger
from agent_layer.config import Settings
from agent_layer.errors import ContextResolutionError, raise_model_error
from agent_layer.retrieval.rewrite import QuestionRewriter
from agent_layer.schemas import ContextResolution, ConversationInput, Message, SessionContext

logger = get_logger("agent.context")


CONTEXT_RESOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你负责招投标会话的指代消解，不回答业务问题。
结合最近消息和已确认实体，将“该公司、那个项目、这条规定、其”等表达改写成独立问题。
只使用输入中已有的实体，不得创造统一社会信用代码、项目 ID、商品 ID 或法规条款。
历史助手回答不是高可信实体来源，除非 entity_state 已确认。
多个候选接近时 ambiguous=true，并给出简短 clarification_question，不得擅自选择。
traces 记录原表达、候选、选择依据和置信度。""",
        ),
        (
            "human",
            "当前问题：\n{question}\n\n最近消息：\n{history}\n\n已确认实体：\n{entity_state}\n\nJSON Schema：\n{schema}",
        ),
    ]
)


class ContextResolver:
    reference_pattern = re.compile(r"该公司|该企业|这个公司|那个公司|该项目|这个项目|那个项目|这条规定|该规定|上述|前述|它|其")

    def __init__(self, rewriter: QuestionRewriter, model: BaseChatModel, settings: Settings):
        self.rewriter = rewriter
        self.settings = settings
        structured = model.with_structured_output(
            ContextResolution,
            method=settings.structured_output_method,
        )
        self.chain = CONTEXT_RESOLUTION_PROMPT | structured

    async def resolve(
        self,
        question: str,
        history_messages: list[Message],
        session_context: SessionContext,
    ) -> ConversationInput:
        rewritten = self.rewriter.rewrite(question)
        if not self._needs_resolution(rewritten, history_messages, session_context):
            return ConversationInput(
                original_question=question,
                standalone_question=rewritten,
                history_messages=history_messages,
                session_context=session_context,
            )

        try:
            resolution = await self.chain.with_retry(
                stop_after_attempt=self.settings.structured_output_retries + 1,
            ).ainvoke(
                {
                    "question": rewritten,
                    "history": self._history_text(history_messages),
                    "entity_state": json.dumps(session_context.entity_state, ensure_ascii=False),
                    "schema": json.dumps(ContextResolution.model_json_schema(), ensure_ascii=False),
                }
            )
        except Exception as exc:
            logger.exception("context resolution failed")
            raise_model_error(exc, ContextResolutionError)
        return ConversationInput(
            original_question=question,
            standalone_question=resolution.standalone_question,
            history_messages=history_messages,
            session_context=session_context,
            resolved_entities=resolution.entities,
            resolution_traces=resolution.traces,
            ambiguous=resolution.ambiguous,
            clarification_question=resolution.clarification_question,
            context_model_calls=1,
        )

    def context_summary(self, conversation: ConversationInput) -> str:
        parts = []
        if conversation.session_context.conversation_summary:
            parts.append(conversation.session_context.conversation_summary[: self.settings.context_message_chars])
        if conversation.resolved_entities:
            parts.append(json.dumps([entity.model_dump() for entity in conversation.resolved_entities], ensure_ascii=False))
        for message in conversation.history_messages[-self.settings.recent_history_messages :]:
            if message.role in {"user", "assistant"} and message.content:
                parts.append(f"{message.role}: {message.content[: self.settings.context_message_chars]}")
        return "\n".join(parts)

    def _needs_resolution(
        self,
        question: str,
        history_messages: list[Message],
        session_context: SessionContext,
    ) -> bool:
        return bool(
            self.reference_pattern.search(question)
            or session_context.entity_state
            or session_context.conversation_summary
            or len(history_messages) > self.settings.recent_history_messages
        )

    def _history_text(self, history_messages: list[Message]) -> str:
        lines = []
        for message in history_messages[-self.settings.recent_history_messages :]:
            if message.role in {"user", "assistant"} and message.content:
                lines.append(f"{message.role}: {message.content[: self.settings.context_message_chars]}")
        return "\n".join(lines)
