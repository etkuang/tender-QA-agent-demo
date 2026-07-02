# coding: utf-8

import re
from typing import Literal

from pydantic import BaseModel

from agent_layer.schemas import Citation, DependencyOutcome, Evidence

ToolName = Literal["rag", "sql", "website", "model_only"]


class ChildTaskWorkflowProfile(BaseModel):
    description: str
    tools_pool: list[ToolName]
    tool_preference: list[list[ToolName]]


def format_workflow_profile(profile: ChildTaskWorkflowProfile) -> str:
    preference = " -> ".join(
        " + ".join(tier)
        for tier in profile.tool_preference
    )
    return "\n".join(
        [
            f"description: {profile.description}",
            f"tools_pool: {', '.join(profile.tools_pool)}",
            f"tool_preference: {preference}",
        ]
    )


def format_dependency_outcomes(dependency_outcomes: list[DependencyOutcome]) -> str:
    if not dependency_outcomes:
        return "无。"
    return "\n".join(
        f"- {outcome.task_id}：{outcome.question}：{outcome.answer}"
        for outcome in dependency_outcomes
    )


def format_evidence(
    evidence: list[Evidence],
    max_length: int = 1200,
    max_chunks: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    blocks = []
    total_length = 0
    selected = evidence[:max_chunks] if max_chunks is not None else evidence
    for index, item in enumerate(selected, 1):
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in item.metadata.items()
            if value not in (None, "", [], {}) and key not in {"raw_content"}
        )
        block = (
            f"[{index}] {item.title}\n"
            f"evidence_id={item.evidence_id}\n"
            f"source_type={item.source_type.value}; published_at={item.published_at}; url={item.url or ''}\n"
            f"metadata={metadata[:500]}\n"
            f"content={item.content[:max_length]}"
        )
        if max_total_chars is not None:
            remaining = max_total_chars - total_length
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
        blocks.append(block)
        total_length += len(block)
    return "\n\n".join(blocks)


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
