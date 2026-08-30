# coding: utf-8
# @Author: Wang Qingkang

from agent_layer_old.schemas import Evidence, RetrievalAssessment
from agent_layer_old.workflows.common import rank_evidence


class PolicySelfRAGPlugin:
    def merge_evidence(self, current: list[Evidence], additions: list[Evidence]) -> list[Evidence]:
        seen = {item.evidence_id for item in current}
        output = list(current)
        for item in additions:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            output.append(item)
        return output

    def next_retrieval_queries(
        self,
        assessment: RetrievalAssessment,
        seen_queries: set[str],
        max_queries: int,
    ) -> list[str]:
        output = []
        for query in assessment.follow_up_queries:
            normalized = query.strip()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            output.append(normalized)
            if len(output) >= max_queries:
                break
        return output

    def select_evidence(self, evidence: list[Evidence], assessment: RetrievalAssessment) -> list[Evidence]:
        ranked = rank_evidence(evidence)
        if not assessment.usable_evidence_ids:
            return ranked
        selected_ids = set(assessment.usable_evidence_ids)
        selected = [item for item in ranked if item.evidence_id in selected_ids]
        return selected or ranked


def next_retrieval_queries(
    assessment: RetrievalAssessment,
    seen_queries: set[str],
    max_queries: int,
) -> list[str]:
    output = []
    for query in assessment.follow_up_queries:
        normalized = query.strip()
        if not normalized or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        output.append(normalized)
        if len(output) >= max_queries:
            break
    return output


def select_evidence(evidence: list[Evidence], assessment: RetrievalAssessment) -> list[Evidence]:
    ranked = rank_evidence(evidence)
    if not assessment.usable_evidence_ids:
        return ranked
    selected_ids = set(assessment.usable_evidence_ids)
    selected = [item for item in ranked if item.evidence_id in selected_ids]
    return selected or ranked