"""Deterministic, read-only remediation for train-to-inference split overlap."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
TRAINING_COMPARISON_SPLITS = ("warmup", "public", "private")


def normalize_question_text(text: str) -> str:
    """Normalize question text for deterministic overlap detection only."""

    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


@dataclass(frozen=True, slots=True)
class TrainingOverlapExclusion:
    """One train case removed from the derived SFT set with explicit reasons."""

    case_id: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize without question or answer text."""

        return {"case_id": self.case_id, "reasons": list(self.reasons)}


def derive_train_overlap_exclusions(
    train_questions_by_id: Mapping[str, str],
    comparison_questions_by_split: Mapping[str, Mapping[str, str]],
) -> tuple[TrainingOverlapExclusion, ...]:
    """Exclude every train ID sharing an ID or normalized question downstream.

    The source split files are never changed.  This deliberately excludes all
    train duplicates of a matching normalized question, rather than retaining an
    arbitrary representative.
    """

    comparisons = {
        split: questions
        for split, questions in comparison_questions_by_split.items()
        if split in TRAINING_COMPARISON_SPLITS
    }
    reasons_by_id: dict[str, list[str]] = {}
    for split in sorted(comparisons):
        questions = comparisons[split]
        other_ids = set(questions)
        other_normalized_questions = {
            normalize_question_text(question) for question in questions.values()
        }
        for case_id, question in train_questions_by_id.items():
            reasons: list[str] = []
            if case_id in other_ids:
                reasons.append(f"id_overlap:{split}")
            if normalize_question_text(question) in other_normalized_questions:
                reasons.append(f"normalized_question_overlap:{split}")
            if reasons:
                reasons_by_id.setdefault(case_id, []).extend(reasons)

    return tuple(
        TrainingOverlapExclusion(case_id=case_id, reasons=tuple(reasons_by_id[case_id]))
        for case_id in sorted(reasons_by_id)
        if reasons_by_id[case_id]
    )


def exclusion_reason_map(
    exclusions: tuple[TrainingOverlapExclusion, ...],
) -> dict[str, str]:
    """Return deterministic, human-readable reasons for dataset exclusions."""

    return {item.case_id: "; ".join(item.reasons) for item in exclusions}


__all__ = [
    "TRAINING_COMPARISON_SPLITS",
    "TrainingOverlapExclusion",
    "derive_train_overlap_exclusions",
    "exclusion_reason_map",
    "normalize_question_text",
]
