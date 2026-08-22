"""Shared query loaders for retrieval scripts (inference-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    SyntheticQueryRecord,
    load_synthetic_records,
)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One query record for BM25/dense retrieval output."""

    query_id: str
    question: str


def load_retrieval_queries_from_json(
    path: str | Path,
    *,
    split: str,
) -> tuple[RetrievalQuery, ...]:
    questions = load_inference_questions(path, split=split)  # type: ignore[arg-type]
    return tuple(
        RetrievalQuery(query_id=question.id, question=question.question)
        for question in questions
    )


def load_retrieval_queries_from_synthetic(
    path: str | Path,
    *,
    source_split: str = "train",
) -> tuple[RetrievalQuery, ...]:
    records = load_synthetic_records(path)
    selected: list[SyntheticQueryRecord] = [
        record
        for record in records
        if not source_split or record.source_split == source_split
    ]
    selected.sort(key=lambda record: record.synthetic_id)
    return tuple(
        RetrievalQuery(query_id=record.synthetic_id, question=record.query)
        for record in selected
    )


__all__ = [
    "RetrievalQuery",
    "load_retrieval_queries_from_json",
    "load_retrieval_queries_from_synthetic",
]
