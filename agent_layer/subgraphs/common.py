# coding: utf-8
# @Author: Wang Qingkang

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from agent_layer.child_task_state_schemas import PrerequisiteAnswer
from agent_layer.config import Settings
from agent_layer.evidence_schemas import WebsiteEvidence
from agent_layer.integrations.evidence_pool import EvidencePool
from agent_layer.integrations.internet_search import InternetSearchGateway
from common.api_contracts.agent_api import Message


@dataclass(frozen=True)
class ChildTaskRuntimeContext:
    settings: Settings
    answer_model: BaseChatModel
    research_model: BaseChatModel
    evidence_pool: EvidencePool
    internet_search_gateway: InternetSearchGateway


def format_blockquote(content: str) -> str:
    lines = content.splitlines() or [""]
    return "\n".join(
        ">" if not line else f"> {line}"
        for line in lines
    )


def format_history_markdown(
    history_messages: list[Message],
) -> str:
    if not history_messages:
        return "无。"

    role_labels = {
        "user": "用户",
        "assistant": "助手",
    }
    blocks = []
    for index, message in enumerate(history_messages, start=1):
        blocks.append(
            f"### 消息 {index}：{role_labels[message.role]}\n\n"
            f"{format_blockquote(message.content)}"
        )
    return "\n\n".join(blocks)


def format_prerequisite_answers_markdown(
    prerequisite_answers: list[PrerequisiteAnswer],
    settings: Settings,
) -> str:
    blocks = []
    total_chars = 0
    for prerequisite in prerequisite_answers:
        question = " ".join(prerequisite.question.split())[
            :settings.prerequisite_question_chars
        ]
        answer = " ".join(prerequisite.answer.split())[
            :settings.prerequisite_answer_chars
        ]
        block = (
            f"### 前置问题 `{prerequisite.task_id}`\n\n"
            f"- 问题：{question}\n"
            f"- 结论：{answer}"
        )
        separator = "\n\n" if blocks else ""
        if (
            total_chars + len(separator) + len(block)
            > settings.prerequisite_context_chars
        ):
            break
        blocks.append(block)
        total_chars += len(separator) + len(block)

    if not blocks:
        return "无。"
    return "\n\n".join(blocks)


def merge_evidence(
    *evidence_groups: list[WebsiteEvidence],
) -> list[WebsiteEvidence]:
    """Keep the first canonical record for each stable evidence id."""
    merged = {}
    for evidence_group in evidence_groups:
        for evidence in evidence_group:
            if evidence.evidence_id not in merged:
                merged[evidence.evidence_id] = evidence
    return list(merged.values())


def format_evidence_markdown(
    evidence: list[WebsiteEvidence],
    settings: Settings,
) -> tuple[list[WebsiteEvidence], str]:
    selected_evidence = []
    blocks = []
    total_chars = 0
    for item in evidence[:settings.evidence_context_limit]:
        title = " ".join(item.title.split())
        updated_at = (
            item.updated_at.date().isoformat()
            if item.updated_at is not None
            else "无"
        )
        content = item.content[:settings.evidence_chunk_chars]
        block = (
            f"### 证据 `{item.evidence_id}`：{title}\n\n"
            f"- 来源类型：{item.source_type.value}\n"
            f"- 更新时间：{updated_at}\n"
            f"- 可靠性说明：{item.reliability_description}\n"
            f"- 链接：{item.url}\n\n"
            f"{format_blockquote(content)}"
        )
        separator = "\n\n" if blocks else ""
        if (
            total_chars + len(separator) + len(block)
            > settings.evidence_context_chars
        ):
            break
        selected_evidence.append(item)
        blocks.append(block)
        total_chars += len(separator) + len(block)

    if not blocks:
        return [], "无。"
    return selected_evidence, "\n\n".join(blocks)


def freshness_requirement(requires_fresh_data: bool) -> str:
    if requires_fresh_data:
        return "需要最新信息"
    return "不需要最新信息"