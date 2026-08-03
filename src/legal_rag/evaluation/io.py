"""Local JSON and JSONL adapters for evaluator inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .alignment import canonical_id
from .models import InputRecord


class InputFormatError(ValueError):
    """Raised when a local reference or prediction file has an invalid shape."""


class _JSONObject(list[tuple[str, Any]]):
    """Retain object pairs so duplicate JSON keys can be reported."""


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> _JSONObject:
    return _JSONObject(pairs)


def _convert_json(value: Any) -> Any:
    if isinstance(value, _JSONObject):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                raise InputFormatError(f"Duplicate JSON object key: {key!r}")
            result[key] = _convert_json(item)
        return result
    if isinstance(value, list):
        return [_convert_json(item) for item in value]
    return value


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            return _convert_json(
                json.load(file_handle, object_pairs_hook=_object_pairs_hook)
            )
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"Invalid JSON in {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputFormatError(f"Input must be UTF-8: {path}") from exc


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            for line_number, raw_line in enumerate(file_handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    decoded = json.loads(raw_line, object_pairs_hook=_object_pairs_hook)
                except json.JSONDecodeError as exc:
                    raise InputFormatError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                records.append(_convert_json(decoded))
    except UnicodeDecodeError as exc:
        raise InputFormatError(f"Input must be UTF-8: {path}") from exc
    return records


def _answer_from_value(value: Any, path: Path, record_id: str, role: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise InputFormatError(
            f"{role} record {record_id!r} in {path} must be a string or object"
        )
    for key in ("answer", "prediction", "reference"):
        answer = value.get(key)
        if isinstance(answer, str):
            return answer
    raise InputFormatError(
        f"{role} record {record_id!r} in {path} has no string answer/prediction field"
    )


def _record_from_object(value: Any, path: Path, role: str) -> InputRecord:
    if not isinstance(value, Mapping):
        raise InputFormatError(f"{role} JSONL entries in {path} must be objects")
    if "id" not in value:
        raise InputFormatError(f"{role} record in {path} is missing 'id'")
    record_id = canonical_id(value["id"])
    return InputRecord(
        id=record_id,
        answer=_answer_from_value(value, path, record_id, role),
    )


def _records_from_payload(payload: Any, path: Path, role: str) -> list[InputRecord]:
    if isinstance(payload, list):
        return [_record_from_object(item, path, role) for item in payload]
    if not isinstance(payload, Mapping):
        raise InputFormatError(f"{role} root in {path} must be an object or list")

    if "records" in payload:
        records = payload["records"]
        if not isinstance(records, list):
            raise InputFormatError(f"{role} field 'records' in {path} must be a list")
        return [_record_from_object(item, path, role) for item in records]

    return [
        InputRecord(
            id=canonical_id(record_id),
            answer=_answer_from_value(value, path, canonical_id(record_id), role),
        )
        for record_id, value in payload.items()
    ]


def load_records(path: str | Path, role: str) -> list[InputRecord]:
    """Load a warmup-style JSON map, JSON record list, or JSONL records.

    JSONL is useful for strict duplicate-ID tests because every record retains
    its source occurrence.  JSON object keys are also checked for duplicates
    during parsing rather than silently using last-wins behavior.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise InputFormatError(f"Input file missing: {input_path}")
    if input_path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return _records_from_payload(_load_jsonl(input_path), input_path, role)
    return _records_from_payload(_load_json(input_path), input_path, role)


def records_from_mapping(records: Mapping[object, Any], role: str) -> list[InputRecord]:
    """Convert an in-memory map for library callers and tests."""

    result: list[InputRecord] = []
    for record_id, value in records.items():
        canonical = canonical_id(record_id)
        result.append(
            InputRecord(
                id=canonical,
                answer=_answer_from_value(value, Path("<memory>"), canonical, role),
            )
        )
    return result


def records_from_iterable(
    records: Iterable[Mapping[str, Any]], role: str
) -> list[InputRecord]:
    """Convert in-memory record objects with explicit ``id`` fields."""

    return [_record_from_object(record, Path("<memory>"), role) for record in records]
