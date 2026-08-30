# coding: utf-8

import re
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from agent_layer_old.data_domain.schemas import DataAnalysisSummary
from agent_layer_old.schemas import Category, Citation, DependencyOutcome, Evidence

ToolName = Literal["rag", "sql", "website"]


class ChildTaskQueryType(BaseModel):
    type_name: str
    description: str
    query_format: str


class ChildTaskToolQuery(BaseModel):
    tool_name: ToolName
    query: str = Field(min_length=1)


class ChildTaskToolSpec(BaseModel):
    tool_name: ToolName
    description: str
    query_format: str


class ChildTaskIntentDecision(BaseModel):
    type_name: str
    terminal_message: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> Self:
        is_terminal = self.type_name in {"clarification", "unsupported"}
        if is_terminal and self.terminal_message is None:
            raise ValueError("Terminal intent decisions require a message")
        if not is_terminal and self.terminal_message is not None:
            raise ValueError(
                "Non-terminal intent decisions must not provide a message"
            )
        return self


class ChildTaskTierQueryPlan(BaseModel):
    tool_queries: list[ChildTaskToolQuery] = Field(min_length=1)


class ChildTaskQueryResult(BaseModel):
    tool_name: ToolName
    original_query: str
    query_result: list[Evidence] = Field(default_factory=list)
    analysis: DataAnalysisSummary | None = None


class ChildTaskWorkflowProfile(BaseModel):
    category: Category
    query_types: list[ChildTaskQueryType]
    tool_preference: list[list[ToolName]]
    model_only_fallback: bool = False


def format_dependency_outcomes(dependency_outcomes: list[DependencyOutcome]) -> str:
    if not dependency_outcomes:
        return "无。"
    return "\n".join(
        f"- {outcome.task_id}：{outcome.question}：{outcome.answer}"
        for outcome in dependency_outcomes
    )


def format_evidence_block(
    evidence: Evidence,
    index: int,
    max_length: int,
) -> str:
    metadata = ", ".join(
        f"{key}={value}"
        for key, value in evidence.metadata.items()
        if value not in (None, "", [], {}) and key not in {"raw_content"}
    )
    return (
        f"[{index}] {evidence.title}\n"
        f"证据 ID={evidence.evidence_id}\n"
        f"来源类型={evidence.source_type.value}; 发布日期={evidence.published_at}; "
        f"网址={evidence.url or ''}\n"
        f"元数据={metadata[:500]}\n"
        f"内容={evidence.content[:max_length]}"
    )


def prepare_evidence_context(
    evidence: list[Evidence],
    max_length: int,
    max_total_chars: int,
) -> tuple[list[Evidence], str]:
    blocks = []
    selected = []
    total_length = 0
    for item in evidence:
        block = format_evidence_block(item, len(selected) + 1, max_length)
        separator = "\n\n" if blocks else ""
        remaining = max_total_chars - total_length - len(separator)
        if len(block) > remaining:
            break
        blocks.append(block)
        selected.append(item)
        total_length += len(separator) + len(block)
    return selected, "\n\n".join(blocks)


def format_query_result_evidence_block(
    evidence: Evidence,
    index: int,
    max_length: int,
) -> str:
    source_identifier = evidence.url or evidence.document_id or evidence.evidence_id
    return (
        f"[{index}]\n"
        f"来源：{evidence.title}；类型={evidence.source_type.value}；"
        f"发布日期={evidence.published_at or '未知'}；标识={source_identifier}\n"
        f"证据：{evidence.content[:max_length]}"
    )


def prepare_query_result_context(
    query_results: list[ChildTaskQueryResult],
    max_length: int,
    max_total_chars: int,
) -> tuple[list[Evidence], str]:
    blocks = []
    selected = []
    total_length = 0

    for query_result in query_results:
        separator = "\n\n" if blocks else ""
        remaining = max_total_chars - total_length - len(separator)
        if remaining <= 0:
            break

        header = (
            f"工具：{query_result.tool_name}\n"
            f"原始查询：{query_result.original_query}\n"
            "查询结果："
        )
        analysis = (
            query_result.analysis.model_dump_json()
            if query_result.analysis is not None
            else "无。"
        )
        result_blocks = []
        exhausted = False

        if query_result.query_result:
            for evidence in query_result.query_result:
                block = format_query_result_evidence_block(
                    evidence,
                    len(selected) + 1,
                    max_length,
                )
                result_text = "\n\n".join([*result_blocks, block])
                candidate = (
                    f"{header}\n{result_text}\n\n"
                    f"确定性分析：{analysis}"
                )
                if len(candidate) > remaining:
                    exhausted = True
                    break
                result_blocks.append(block)
                selected.append(evidence)
            result_text = (
                "\n\n".join(result_blocks)
                if result_blocks
                else "上下文预算不足，未纳入证据。"
            )
        else:
            result_text = "无返回结果。"

        block = (
            f"{header}\n{result_text}\n\n"
            f"确定性分析：{analysis}"
        )
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        total_length += len(separator) + len(block)
        if exhausted or len(block) == remaining:
            break

    text = "\n\n".join(blocks) if blocks else "无外部查询结果。"
    return selected, text


def build_citations(evidence: list[Evidence]) -> list[Citation]:
    output = []
    for index, item in enumerate(evidence, 1):
        output.append(
            Citation(
                citation_id=f"source-{index}",
                evidence_id=item.evidence_id,
                label=item.title,
                url=item.url,
            )
        )
    return output


def rank_evidence(evidence: list[Evidence]) -> list[Evidence]:
    output = []
    seen = set()
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.authority_level,
            item.freshness_level,
            item.score if item.score is not None else 0,
        ),
        reverse=True,
    )
    for item in ordered:
        content_hash = item.metadata.get("content_hash")
        key = content_hash or item.url or item.evidence_id
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def ensure_source_section(answer: str, citations: list[Citation]) -> str:
    if not citations or "来源" in answer[-200:]:
        return answer
    lines = ["", "来源："]
    for index, citation in enumerate(citations, 1):
        if citation.url:
            lines.append(f"[{index}] {citation.label}：{citation.url}")
        else:
            lines.append(f"[{index}] {citation.label}")
    return f"{answer.rstrip()}\n" + "\n".join(lines)


def citations_are_valid(answer: str, citation_count: int) -> bool:
    markers = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    if citation_count == 0:
        return not markers
    if not markers:
        return False
    return all(1 <= marker <= citation_count for marker in markers)
