"""Strict ID-based prediction/reference alignment."""

from __future__ import annotations

from collections.abc import Iterable

from .models import AlignedRecord, InputRecord


class AlignmentError(ValueError):
    """Raised when prediction and reference IDs cannot form a complete match."""


class DuplicateIDError(AlignmentError):
    """Raised when one input contains an ID more than once."""


def canonical_id(value: object) -> str:
    """Convert an input ID to the contract's internal string representation."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AlignmentError(
            f"ID must be a string or integer, got {type(value).__name__}"
        )
    result = str(value)
    if not result:
        raise AlignmentError("ID must not be empty")
    return result


def _index_records(records: Iterable[InputRecord], role: str) -> dict[str, InputRecord]:
    indexed: dict[str, InputRecord] = {}
    duplicates: list[str] = []
    for record in records:
        record_id = canonical_id(record.id)
        if not isinstance(record.answer, str):
            raise AlignmentError(f"{role} answer for ID {record_id!r} must be a string")
        if record_id in indexed:
            duplicates.append(record_id)
            continue
        indexed[record_id] = InputRecord(id=record_id, answer=record.answer)

    if duplicates:
        duplicate_ids = ", ".join(sorted(set(duplicates)))
        raise DuplicateIDError(f"{role} contains duplicate ID(s): {duplicate_ids}")
    return indexed


def align_records(
    references: Iterable[InputRecord], predictions: Iterable[InputRecord]
) -> tuple[AlignedRecord, ...]:
    """Align complete record sets by canonical ID in deterministic order.

    Missing, extra, and duplicate IDs are hard errors.  The evaluator never
    scores the intersection of two incomplete sets.
    """

    reference_by_id = _index_records(references, "Reference")
    prediction_by_id = _index_records(predictions, "Prediction")

    reference_ids = set(reference_by_id)
    prediction_ids = set(prediction_by_id)
    missing = sorted(reference_ids - prediction_ids)
    extra = sorted(prediction_ids - reference_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing prediction ID(s): {', '.join(missing)}")
        if extra:
            details.append(f"extra prediction ID(s): {', '.join(extra)}")
        raise AlignmentError(
            "Prediction/reference alignment failed: " + "; ".join(details)
        )

    return tuple(
        AlignedRecord(
            id=record_id,
            reference=reference_by_id[record_id].answer,
            prediction=prediction_by_id[record_id].answer,
        )
        for record_id in sorted(reference_ids)
    )
