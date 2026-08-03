"""Read-only loader for the observed competition question-map schema."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import InferenceQuestion, LegalQuestion


class QuestionLoadError(ValueError):
    """Raised when a question source cannot satisfy the observed schema."""


@dataclass(frozen=True)
class QuestionSourceStats:
    """Content-free diagnostics from a raw question source."""

    record_count: int
    answer_available_count: int
    answer_missing_count: int
    duplicate_id_count: int
    blank_question_count: int
    blank_answer_count: int
    question_lengths: tuple[int, ...]
    answer_lengths: tuple[int, ...]


class _ObjectPairs(list[tuple[str, Any]]):
    """Keep JSON object pairs so duplicate source keys are not lost."""


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> _ObjectPairs:
    return _ObjectPairs(pairs)


def _location(path: Path, record_key: str | None = None) -> str:
    if record_key is None:
        return str(path)
    return f"{path}: record key {record_key!r}"


def _read_json(path: Path) -> _ObjectPairs:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            payload = json.load(file_handle, object_pairs_hook=_object_pairs_hook)
    except FileNotFoundError as exc:
        raise QuestionLoadError(f"Source file missing: {path}") from exc
    except UnicodeDecodeError as exc:
        raise QuestionLoadError(f"Source file must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise QuestionLoadError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, _ObjectPairs):
        raise QuestionLoadError(
            f"{path}: top-level schema must be a JSON object/map; "
            f"got {type(payload).__name__}"
        )
    return payload


def _record_fields(path: Path, record_key: str, value: Any) -> dict[str, Any]:
    location = _location(path, record_key)
    if not isinstance(value, _ObjectPairs):
        raise QuestionLoadError(
            f"{location}: record value must be a JSON object with question/answer"
        )

    fields: dict[str, Any] = {}
    for field_name, field_value in value:
        if field_name in fields:
            raise QuestionLoadError(
                f"{location}: duplicate record field {field_name!r}"
            )
        fields[field_name] = field_value

    expected_fields = {"question", "answer"}
    missing_fields = expected_fields - set(fields)
    # ``answer`` is the only optional field for inference-safe inputs.  The
    # observed warm-up records contain it, while official inference files may
    # omit gold answers under the frozen contract.
    if "question" in missing_fields:
        raise QuestionLoadError(f"{location}: missing required field 'question'")
    extra_fields = set(fields) - expected_fields
    if extra_fields:
        raise QuestionLoadError(
            f"{location}: unexpected field(s): {', '.join(sorted(extra_fields))}"
        )
    if not isinstance(fields["question"], str):
        raise QuestionLoadError(f"{location}: field 'question' must be a string")
    if (
        "answer" in fields
        and fields["answer"] is not None
        and not isinstance(fields["answer"], str)
    ):
        raise QuestionLoadError(
            f"{location}: field 'answer' must be a string when present"
        )
    return fields


def _build_question(
    path: Path, record_key: str, fields: dict[str, Any], split: str
) -> LegalQuestion:
    try:
        return LegalQuestion(
            id=record_key,
            question=fields["question"],
            answer=fields.get("answer"),
            split=split,
        )
    except ValidationError as exc:
        raise QuestionLoadError(
            f"{_location(path, record_key)}: invalid question record: {exc}"
        ) from exc


def _validate_split(path: Path, split: str) -> None:
    if not isinstance(split, str) or not split.strip():
        raise QuestionLoadError(f"{path}: split label must be a non-blank string")


def inspect_questions(path: str | Path) -> QuestionSourceStats:
    """Count diagnostics without returning question or answer text.

    This inspection shares the production JSON parser with :func:`load_questions` but
    deliberately does not treat malformed records as valid. The loader remains the
    authoritative critical schema check.
    """

    payload = _read_json(Path(path))
    seen_ids: set[str] = set()
    duplicate_id_count = 0
    answer_available_count = 0
    answer_missing_count = 0
    blank_question_count = 0
    blank_answer_count = 0
    question_lengths: list[int] = []
    answer_lengths: list[int] = []

    for record_key, value in payload:
        if record_key in seen_ids:
            duplicate_id_count += 1
        seen_ids.add(record_key)
        if not isinstance(value, _ObjectPairs):
            continue

        fields: dict[str, Any] = {}
        for field_name, field_value in value:
            fields.setdefault(field_name, field_value)

        question = fields.get("question")
        if isinstance(question, str):
            question_lengths.append(len(question))
            if not question.strip():
                blank_question_count += 1

        answer = fields.get("answer")
        if isinstance(answer, str):
            answer_available_count += 1
            answer_lengths.append(len(answer))
            if not answer.strip():
                blank_answer_count += 1
        else:
            answer_missing_count += 1

    return QuestionSourceStats(
        record_count=len(payload),
        answer_available_count=answer_available_count,
        answer_missing_count=answer_missing_count,
        duplicate_id_count=duplicate_id_count,
        blank_question_count=blank_question_count,
        blank_answer_count=blank_answer_count,
        question_lengths=tuple(question_lengths),
        answer_lengths=tuple(answer_lengths),
    )


def load_questions(path: str | Path, *, split: str) -> tuple[LegalQuestion, ...]:
    """Load the observed question map without modifying the source file.

    The source is expected to be a UTF-8 JSON object whose keys are question
    IDs and whose values are nested objects with exactly ``question`` and
    ``answer`` fields.  ``answer`` may be omitted for inference-only input.
    Returned records are sorted by their canonical string IDs.
    """

    source_path = Path(path)
    _validate_split(source_path, split)
    payload = _read_json(source_path)

    records: list[LegalQuestion] = []
    seen_ids: set[str] = set()
    for record_key, value in payload:
        # JSON object keys are strings.  Keep their exact spelling; canonical
        # ID normalization is representation as string, not numeric rewriting.
        canonical_id = str(record_key)
        if canonical_id in seen_ids:
            raise QuestionLoadError(
                f"{_location(source_path, canonical_id)}: duplicate question ID"
            )
        seen_ids.add(canonical_id)
        fields = _record_fields(source_path, canonical_id, value)
        records.append(_build_question(source_path, canonical_id, fields, split))

    return tuple(sorted(records, key=lambda record: record.id))


def inference_view(
    records: Iterable[LegalQuestion],
) -> tuple[InferenceQuestion, ...]:
    """Return deterministic question-only records with gold answers removed."""

    return tuple(record.inference_view() for record in records)
