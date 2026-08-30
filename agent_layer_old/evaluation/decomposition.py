# coding: utf-8

from pydantic import BaseModel, Field

from agent_layer_old.question_decomposition.chain import QuestionDecomposer
from agent_layer_old.schemas import Category


class DecompositionSample(BaseModel):
    sample_id: str
    question: str
    expected_categories: list[Category] = Field(default_factory=list)
    history: str = ""


class DecompositionEvaluation(BaseModel):
    total: int
    exact_match_accuracy: float
    macro_f1: float
    failures: list[str]


async def evaluate_decomposition(
    decomposer: QuestionDecomposer,
    samples: list[DecompositionSample],
) -> DecompositionEvaluation:
    labels = list(Category)
    counts = {label: {"true_positive": 0, "false_positive": 0, "false_negative": 0} for label in labels}
    failures = []
    exact_matches = 0
    for sample in samples:
        tasks = await decomposer.decompose(sample.question, sample.history)
        expected = set(sample.expected_categories)
        predicted = {task.category for task in tasks}
        if predicted == expected:
            exact_matches += 1
        else:
            failures.append(sample.sample_id)
        for label in labels:
            if label in expected and label in predicted:
                counts[label]["true_positive"] += 1
            elif label not in expected and label in predicted:
                counts[label]["false_positive"] += 1
            elif label in expected and label not in predicted:
                counts[label]["false_negative"] += 1
    f1_scores = []
    for label in labels:
        true_positive = counts[label]["true_positive"]
        false_positive = counts[label]["false_positive"]
        false_negative = counts[label]["false_negative"]
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    total = len(samples)
    return DecompositionEvaluation(
        total=total,
        exact_match_accuracy=exact_matches / total if total else 0,
        macro_f1=sum(f1_scores) / len(f1_scores),
        failures=failures,
    )
