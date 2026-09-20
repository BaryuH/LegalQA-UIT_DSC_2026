"""Typed boundaries for the auxiliary extractive-reader baselines."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from ..schemas import CanonicalID, DomainModel, NonBlankText


class ReaderInferenceCase(DomainModel):
    """Inference-safe case; a reference answer cannot cross this boundary."""

    id: CanonicalID
    question: NonBlankText
    context: NonBlankText


class ReaderCase(ReaderInferenceCase):
    """Evaluation/training view; convert to ``inference_view`` before inference."""

    answer: str | None = None

    def inference_view(self) -> ReaderInferenceCase:
        return ReaderInferenceCase(
            id=self.id,
            question=self.question,
            context=self.context,
        )


class ReaderSpan(DomainModel):
    """One raw extractive span emitted by a reader backend."""

    answer: str
    confidence: float
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    impossible: bool = False


class ReaderCandidate(DomainModel):
    """Context candidate passed to the same reader checkpoint."""

    source_case_id: CanonicalID
    origin: Literal["original", "train_retrieval"]
    context: NonBlankText
    candidate_order: int = Field(ge=0)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None


class ReaderPrediction(DomainModel):
    """Gold-free final prediction and provenance for one reader case."""

    id: CanonicalID
    answer: str
    method: Literal["finetuned_reader", "tuned_bm25_reader"]
    confidence: float
    source_case_id: CanonicalID
    source_origin: Literal["original", "train_retrieval"]
    model: NonBlankText
    model_version: NonBlankText


@runtime_checkable
class ExtractiveReader(Protocol):
    """Backend contract shared by both reader profiles."""

    @property
    def model(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def predict(
        self,
        *,
        question: str,
        context: str,
        case_id: str,
    ) -> ReaderSpan: ...


__all__ = [
    "ExtractiveReader",
    "ReaderCandidate",
    "ReaderCase",
    "ReaderInferenceCase",
    "ReaderPrediction",
    "ReaderSpan",
]
