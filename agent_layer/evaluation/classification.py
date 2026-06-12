# coding: utf-8

from pydantic import BaseModel

from agent_layer.classification.chain import QuestionClassifier
from agent_layer.schemas import Category


class ClassificationSample(BaseModel):
    sample_id: str
    question: str
    expected_category: Category
    context_summary: str = ""


class ClassificationEvaluation(BaseModel):
    total: int
    accuracy: float
    macro_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    failures: list[str]


async def evaluate_classification(
    classifier: QuestionClassifier,
    samples: list[ClassificationSample],
) -> ClassificationEvaluation:
    labels = [category.value for category in Category]
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    failures = []
    correct = 0
    for sample in samples:
        result = await classifier.classify(sample.question, sample.context_summary)
        matrix[sample.expected_category.value][result.category.value] += 1
        if result.category == sample.expected_category:
            correct += 1
        else:
            failures.append(sample.sample_id)
    f1_scores = []
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[other][label] for other in labels if other != label)
        false_negative = sum(matrix[label][other] for other in labels if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    total = len(samples)
    return ClassificationEvaluation(
        total=total,
        accuracy=correct / total if total else 0,
        macro_f1=sum(f1_scores) / len(f1_scores),
        confusion_matrix=matrix,
        failures=failures,
    )
