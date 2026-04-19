"""RAGAS-based evaluation: faithfulness, answer correctness, context precision/recall."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypedDict


class EvalSample(TypedDict):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def evaluate_dataset(samples: Iterable[EvalSample]) -> Any:
    """Run RAGAS metrics on (question, answer, contexts, ground_truth) tuples."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = list(samples)
    ds = Dataset.from_list(rows)
    return evaluate(
        ds,
        metrics=[faithfulness, answer_correctness, context_precision, context_recall],
    )
