"""Local exact-match and whitespace-token F1 for reader experiments."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .types import ReaderCase, ReaderPrediction


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class ReaderMetrics:
    exact_match: float
    token_f1: float
    count: int


def evaluate_reader_predictions(
    predictions: Sequence[ReaderPrediction],
    references: Sequence[ReaderCase],
) -> ReaderMetrics:
    """Join by ID inside the evaluation boundary and compute deterministic metrics."""

    if not references:
        raise ValueError("Reader evaluation requires at least one reference")
    prediction_by_id: Mapping[str, ReaderPrediction] = {
        prediction.id: prediction for prediction in predictions
    }
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Reader predictions contain duplicate IDs")
    if {case.id for case in references} != set(prediction_by_id):
        raise ValueError("Reader prediction/reference IDs must match exactly")
    exact_values: list[float] = []
    f1_values: list[float] = []
    for reference in references:
        if reference.answer is None:
            raise ValueError("Reader evaluation requires reference answers")
        predicted = _normalize(prediction_by_id[reference.id].answer)
        gold = _normalize(reference.answer)
        exact_values.append(float(predicted == gold))
        predicted_tokens = predicted.split()
        gold_tokens = gold.split()
        if not predicted_tokens and not gold_tokens:
            f1_values.append(1.0)
            continue
        overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            f1_values.append(0.0)
            continue
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(gold_tokens)
        f1_values.append(2 * precision * recall / (precision + recall))
    count = len(references)
    return ReaderMetrics(
        exact_match=sum(exact_values) / count,
        token_f1=sum(f1_values) / count,
        count=count,
    )


__all__ = ["ReaderMetrics", "evaluate_reader_predictions"]
