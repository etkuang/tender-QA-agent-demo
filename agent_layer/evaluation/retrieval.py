# coding: utf-8

from pydantic import BaseModel, Field

from agent_layer.retrieval.pipeline import RetrievalPipeline


class RetrievalSample(BaseModel):
    sample_id: str
    question: str
    expected_document_ids: list[str] = Field(default_factory=list)


class RetrievalEvaluation(BaseModel):
    total: int
    recall_at_k: float
    mean_reciprocal_rank: float
    failures: list[str]


async def evaluate_retrieval(
    pipeline: RetrievalPipeline,
    samples: list[RetrievalSample],
) -> RetrievalEvaluation:
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    failures = []
    for sample in samples:
        result = await pipeline.retrieve_policy(sample.question)
        returned_ids = [evidence.document_id for evidence in result.evidence]
        expected = set(sample.expected_document_ids)
        hits = expected.intersection(returned_ids)
        recall_sum += len(hits) / len(expected) if expected else 0
        reciprocal_rank = 0.0
        for rank, document_id in enumerate(returned_ids, 1):
            if document_id in expected:
                reciprocal_rank = 1 / rank
                break
        reciprocal_rank_sum += reciprocal_rank
        if not hits:
            failures.append(sample.sample_id)
    total = len(samples)
    return RetrievalEvaluation(
        total=total,
        recall_at_k=recall_sum / total if total else 0,
        mean_reciprocal_rank=reciprocal_rank_sum / total if total else 0,
        failures=failures,
    )
