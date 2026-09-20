"""Fail-closed serializers for the fixed official submission contract.

The public official path is the fixed ``submission.zip``/``submission.json``
object-by-question-ID contract.  The earlier descriptor API remains available
under explicit ``*_with_schema`` compatibility names for existing internal tests;
the CLI never uses that legacy adapter.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

SubmissionContainer = Literal["list", "dict"]
SubmissionDictMode = Literal["answer_map", "record_map"]
SubmissionIDType = Literal["string", "integer"]
SubmissionOrdering = Literal["target", "id"]

_SCHEMA_FIELDS = {
    "format",
    "container",
    "fields",
    "id_field",
    "answer_field",
    "id_type",
    "ordering",
    "dict_mode",
}
_PREDICTION_FIELDS = {
    "id",
    "answer",
    "raw_answer",
    "cleaned_answer",
    "method",
    "status",
    "error_code",
    "fallback_reason",
}
_INTERNAL_FIELDS = {
    "evidence",
    "gold",
    "gold_answer",
    "metadata",
    "method",
    "prediction",
    "reference",
    "reference_answer",
    "retrieval",
    "score",
    "scores",
    "status",
}
_INTEGER_ID = re.compile(r"^(?:0|[1-9][0-9]*)$")


class SubmissionError(ValueError):
    """Raised when a submission cannot satisfy its approved contract."""

    def __init__(self, message: str, *, code: str = "SUBMISSION_INVALID_JSON") -> None:
        super().__init__(message)
        self.code = code


SUBMISSION_ERROR_CODES = (
    "SUBMISSION_FILE_NOT_FOUND",
    "SUBMISSION_WRONG_FILENAME",
    "SUBMISSION_INVALID_UTF8",
    "SUBMISSION_INVALID_JSON",
    "SUBMISSION_TOP_LEVEL_NOT_OBJECT",
    "SUBMISSION_INVALID_QUESTION_ID",
    "SUBMISSION_DUPLICATE_QUESTION_ID",
    "SUBMISSION_MISSING_QUESTION_ID",
    "SUBMISSION_EXTRA_QUESTION_ID",
    "SUBMISSION_VALUE_NOT_OBJECT",
    "SUBMISSION_MISSING_ANSWER",
    "SUBMISSION_ANSWER_NOT_STRING",
    "SUBMISSION_EXTRA_FIELDS",
    "SUBMISSION_EMPTY_ANSWER",
    "SUBMISSION_ZIP_INVALID",
    "SUBMISSION_ZIP_WRONG_LAYOUT",
    "SUBMISSION_ZIP_EXTRA_MEMBER",
    "SUBMISSION_ZIP_MISSING_JSON",
)


@dataclass(frozen=True, slots=True)
class SubmissionSpec:
    """Explicit schema descriptor supplied by the competition owner."""

    container: SubmissionContainer
    fields: tuple[str, ...]
    id_field: str
    answer_field: str
    id_type: SubmissionIDType
    ordering: SubmissionOrdering
    dict_mode: SubmissionDictMode = "answer_map"

    def __post_init__(self) -> None:
        if self.container not in {"list", "dict"}:
            raise SubmissionError("Submission container must be 'list' or 'dict'")
        if self.id_type not in {"string", "integer"}:
            raise SubmissionError("Submission id_type must be 'string' or 'integer'")
        if self.ordering not in {"target", "id"}:
            raise SubmissionError("Submission ordering must be 'target' or 'id'")
        if self.dict_mode not in {"answer_map", "record_map"}:
            raise SubmissionError(
                "Submission dict_mode must be 'answer_map' or 'record_map'"
            )
        field_names = tuple(self.fields)
        if not field_names:
            raise SubmissionError("Submission schema must declare non-empty fields")
        if len(set(field_names)) != len(field_names):
            raise SubmissionError("Submission schema fields must be unique")
        if any(not field.strip() for field in field_names):
            raise SubmissionError("Submission schema field names must not be blank")
        if self.id_field not in field_names:
            raise SubmissionError("Submission schema fields must include id_field")
        if self.answer_field not in field_names:
            raise SubmissionError("Submission schema fields must include answer_field")
        if self.id_field == self.answer_field:
            raise SubmissionError("id_field and answer_field must be different")
        for field_name in field_names:
            if field_name.casefold() in _INTERNAL_FIELDS:
                raise SubmissionError(
                    f"Submission schema contains internal field: {field_name}"
                )
            if field_name not in {self.id_field, self.answer_field}:
                raise SubmissionError(
                    "The writer only supports fields sourced from prediction id or "
                    f"answer; unsupported field: {field_name}"
                )
        if self.container == "list" and self.dict_mode != "answer_map":
            raise SubmissionError("dict_mode is only valid for dict submissions")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SubmissionSpec:
        """Build a spec from a strict JSON descriptor."""

        unknown = set(value) - _SCHEMA_FIELDS
        if unknown:
            raise SubmissionError(
                "Unknown submission schema field(s): " + ", ".join(sorted(unknown))
            )
        container_value = value.get("container", value.get("format"))
        if "container" in value and "format" in value:
            if value["container"] != value["format"]:
                raise SubmissionError("Submission format and container must agree")
        required = {"fields", "id_field", "answer_field", "id_type", "ordering"}
        missing = required - set(value)
        if "container" not in value and "format" not in value:
            missing.add("container")
        if missing:
            raise SubmissionError(
                "Submission schema is missing required field(s): "
                + ", ".join(sorted(missing))
            )
        if not isinstance(container_value, str):
            raise SubmissionError("Submission schema container must be a string")
        fields_value = value["fields"]
        if not isinstance(fields_value, list) or not all(
            isinstance(field, str) for field in fields_value
        ):
            raise SubmissionError("Submission schema fields must be a JSON string list")
        string_fields = ("id_field", "answer_field", "id_type", "ordering")
        for field_name in string_fields:
            if not isinstance(value[field_name], str):
                raise SubmissionError(
                    f"Submission schema {field_name} must be a string"
                )
        dict_mode = value.get("dict_mode", "answer_map")
        if not isinstance(dict_mode, str):
            raise SubmissionError("Submission schema dict_mode must be a string")
        return cls(
            container=container_value,  # type: ignore[arg-type]
            fields=tuple(fields_value),
            id_field=value["id_field"],
            answer_field=value["answer_field"],
            id_type=value["id_type"],  # type: ignore[arg-type]
            ordering=value["ordering"],  # type: ignore[arg-type]
            dict_mode=dict_mode,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    """Content-free result of validating one submission payload."""

    record_count: int
    ordered_ids: tuple[str, ...]


class _JSONObject(list[tuple[str, Any]]):
    """Retain JSON object pairs so duplicate keys cannot be last-write-wins."""


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> _JSONObject:
    return _JSONObject(pairs)


def _convert_json(value: Any, *, path: Path) -> Any:
    if isinstance(value, _JSONObject):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                raise SubmissionError(f"Duplicate JSON object key in {path}: {key!r}")
            result[key] = _convert_json(item, path=path)
        return result
    if isinstance(value, list):
        return [_convert_json(item, path=path) for item in value]
    return value


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            decoded = json.load(file_handle, object_pairs_hook=_object_pairs_hook)
    except FileNotFoundError as exc:
        raise SubmissionError(f"File missing: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"File must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"Invalid JSON in {path}: {exc}") from exc
    return _convert_json(decoded, path=path)


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            for line_number, raw_line in enumerate(file_handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    decoded = json.loads(
                        raw_line,
                        object_pairs_hook=_object_pairs_hook,
                    )
                except json.JSONDecodeError as exc:
                    raise SubmissionError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                records.append(_convert_json(decoded, path=path))
    except FileNotFoundError as exc:
        raise SubmissionError(f"File missing: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"File must be UTF-8: {path}") from exc
    return records


def load_submission_spec(path: str | Path) -> SubmissionSpec:
    """Load an explicit schema descriptor; never infer one from input data."""

    schema_path = Path(path)
    payload = _load_json(schema_path)
    if not isinstance(payload, Mapping):
        raise SubmissionError(
            f"Submission schema root must be an object: {schema_path}"
        )
    return SubmissionSpec.from_mapping(payload)


def _canonical_id(value: object, *, location: str) -> str:
    if value is None or isinstance(value, bool):
        raise SubmissionError(f"{location}: ID must not be null or boolean")
    identifier = str(value)
    if not identifier.strip():
        raise SubmissionError(f"{location}: ID must not be blank")
    return identifier


def _validate_answer(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionError(f"{location}: answer must be a non-blank string")
    return value


def _prediction_row(value: object, *, location: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise SubmissionError(f"{location}: prediction must be an object")
    missing = {"id", "answer"} - set(value)
    if missing:
        raise SubmissionError(
            f"{location}: prediction is missing required field(s): "
            + ", ".join(sorted(missing))
        )
    extra = set(value) - _PREDICTION_FIELDS
    if extra:
        raise SubmissionError(
            f"{location}: unsupported prediction field(s): {', '.join(sorted(extra))}"
        )
    identifier = _canonical_id(value["id"], location=f"{location}.id")
    answer = _validate_answer(value["answer"], location=f"{location}.answer")
    return identifier, answer


def load_predictions(path: str | Path) -> tuple[tuple[str, str], ...]:
    """Load prediction IDs and answers while ignoring only known internal fields."""

    prediction_path = Path(path)
    raw_records = (
        _load_jsonl(prediction_path)
        if prediction_path.suffix.casefold() in {".jsonl", ".ndjson"}
        else _load_json(prediction_path)
    )
    if isinstance(raw_records, Mapping):
        if "records" in raw_records:
            if set(raw_records) != {"records"}:
                raise SubmissionError(
                    "Predictions object may contain only the 'records' field"
                )
            raw_records = raw_records["records"]
        else:
            raise SubmissionError(
                "Predictions input must be JSONL, a list, or an object with 'records'"
            )
    if not isinstance(raw_records, list):
        raise SubmissionError("Predictions input must contain a list of records")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(raw_records):
        identifier, answer = _official_prediction_row(
            record,
            location=f"prediction[{index}]",
        )
        if identifier in seen:
            raise SubmissionError(f"Duplicate prediction ID: {identifier!r}")
        seen.add(identifier)
        records.append((identifier, answer))
    return tuple(records)


def _target_id(value: object, *, location: str) -> str:
    if isinstance(value, Mapping):
        if "id" not in value:
            raise SubmissionError(f"{location}: target record must contain 'id'")
        value = value["id"]
    return _canonical_id(value, location=location)


def load_target_ids(path: str | Path) -> tuple[str, ...]:
    """Load target IDs without reading or emitting target questions or answers."""

    target_path = Path(path)
    raw_payload = (
        _load_jsonl(target_path)
        if target_path.suffix.casefold() in {".jsonl", ".ndjson"}
        else _load_json(target_path)
    )
    if isinstance(raw_payload, Mapping):
        if "ids" in raw_payload:
            if set(raw_payload) != {"ids"}:
                raise SubmissionError(
                    "Target ID object may contain only the 'ids' field"
                )
            raw_ids = raw_payload["ids"]
        elif "records" in raw_payload:
            if set(raw_payload) != {"records"}:
                raise SubmissionError(
                    "Target record object may contain only the 'records' field"
                )
            raw_ids = raw_payload["records"]
        else:
            # The observed warm-up/question shape is an ID-keyed map.  Only keys
            # are consumed; nested question/answer values never enter this path.
            raw_ids = list(raw_payload)
    else:
        raw_ids = raw_payload
    if not isinstance(raw_ids, list):
        raise SubmissionError("Target IDs input must be a JSON list or ID-keyed object")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_ids):
        identifier = _target_id(value, location=f"target[{index}]")
        if identifier in seen:
            raise SubmissionError(f"Duplicate target ID: {identifier!r}")
        seen.add(identifier)
        result.append(identifier)
    if not result:
        raise SubmissionError("Target ID set must not be empty")
    return tuple(result)


def _ordered_ids(
    target_ids: tuple[str, ...],
    prediction_ids: set[str],
    ordering: SubmissionOrdering,
) -> tuple[str, ...]:
    target_set = set(target_ids)
    missing = target_set - prediction_ids
    extra = prediction_ids - target_set
    if missing or extra:
        raise SubmissionError(
            "Prediction IDs do not exactly cover target IDs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if ordering == "target":
        return target_ids
    return tuple(sorted(target_ids))


def _serialize_id(
    identifier: str, id_type: SubmissionIDType, *, location: str
) -> str | int:
    if id_type == "string":
        return identifier
    if not _INTEGER_ID.fullmatch(identifier):
        raise SubmissionError(
            f"{location}: ID {identifier!r} is not representable as an official integer"
        )
    return int(identifier)


def _record(
    identifier: str,
    answer: str,
    spec: SubmissionSpec,
    *,
    location: str,
) -> dict[str, str | int]:
    values: dict[str, str | int] = {}
    for field_name in spec.fields:
        if field_name == spec.id_field:
            values[field_name] = _serialize_id(
                identifier, spec.id_type, location=location
            )
        elif field_name == spec.answer_field:
            values[field_name] = answer
        else:  # guarded by SubmissionSpec.__post_init__
            raise SubmissionError(
                f"{location}: unsupported output field {field_name!r}"
            )
    return values


def build_submission(
    predictions: Iterable[Mapping[str, object]],
    target_ids: Iterable[object],
    spec: SubmissionSpec,
) -> list[dict[str, str | int]] | dict[str, Any]:
    """Build a validated, deterministic submission payload in memory."""

    parsed_predictions: list[tuple[str, str]] = []
    seen_predictions: set[str] = set()
    for index, prediction in enumerate(predictions):
        identifier, answer = _prediction_row(
            prediction, location=f"prediction[{index}]"
        )
        if identifier in seen_predictions:
            raise SubmissionError(f"Duplicate prediction ID: {identifier!r}")
        seen_predictions.add(identifier)
        parsed_predictions.append((identifier, answer))

    parsed_targets: list[str] = []
    seen_targets: set[str] = set()
    for index, target in enumerate(target_ids):
        identifier = _target_id(target, location=f"target[{index}]")
        if identifier in seen_targets:
            raise SubmissionError(f"Duplicate target ID: {identifier!r}")
        seen_targets.add(identifier)
        parsed_targets.append(identifier)
    if not parsed_targets:
        raise SubmissionError("Target ID set must not be empty")

    ordered_ids = _ordered_ids(
        tuple(parsed_targets),
        seen_predictions,
        spec.ordering,
    )
    answers = dict(parsed_predictions)
    if spec.container == "list":
        payload: list[dict[str, str | int]] | dict[str, Any] = [
            _record(
                identifier,
                answers[identifier],
                spec,
                location=f"submission[{index}]",
            )
            for index, identifier in enumerate(ordered_ids)
        ]
    elif spec.dict_mode == "answer_map":
        if spec.id_type != "string":
            raise SubmissionError("JSON answer-map keys require id_type='string'")
        payload = {identifier: answers[identifier] for identifier in ordered_ids}
    else:
        if spec.id_type != "string":
            raise SubmissionError("JSON record-map keys require id_type='string'")
        payload = {
            identifier: {
                field_name: value
                for field_name, value in _record(
                    identifier,
                    answers[identifier],
                    spec,
                    location=f"submission[{index}]",
                ).items()
                if field_name != spec.id_field
            }
            for index, identifier in enumerate(ordered_ids)
        }
    validate_submission(payload, target_ids=tuple(parsed_targets), spec=spec)
    return payload


def _validate_record(
    value: object,
    identifier: str,
    spec: SubmissionSpec,
    *,
    location: str,
) -> None:
    if not isinstance(value, Mapping):
        raise SubmissionError(f"{location}: submission record must be an object")
    if set(value) != set(spec.fields):
        raise SubmissionError(
            f"{location}: fields do not match official schema; "
            f"expected={list(spec.fields)}, actual={sorted(value)}"
        )
    output_identifier = value[spec.id_field]
    expected_identifier = _serialize_id(identifier, spec.id_type, location=location)
    if output_identifier != expected_identifier:
        raise SubmissionError(f"{location}: ID does not match target ID {identifier!r}")
    _validate_answer(
        value[spec.answer_field], location=f"{location}.{spec.answer_field}"
    )


def validate_submission(
    payload: object,
    *,
    target_ids: Iterable[object],
    spec: SubmissionSpec,
) -> SubmissionValidation:
    """Validate exact fields, coverage, uniqueness, order, and answers."""

    parsed_targets = tuple(
        _target_id(value, location=f"target[{index}]")
        for index, value in enumerate(target_ids)
    )
    if not parsed_targets or len(set(parsed_targets)) != len(parsed_targets):
        raise SubmissionError("Target ID set must be non-empty and unique")

    if spec.container == "list":
        if not isinstance(payload, list):
            raise SubmissionError("Submission root must be a JSON list")
        ordered_ids: list[str] = []
        for index, value in enumerate(payload):
            if not isinstance(value, Mapping) or spec.id_field not in value:
                raise SubmissionError(
                    f"submission[{index}]: missing official ID field {spec.id_field!r}"
                )
            identifier = _canonical_id(
                value[spec.id_field],
                location=f"submission[{index}].{spec.id_field}",
            )
            if identifier in ordered_ids:
                raise SubmissionError(f"Duplicate submission ID: {identifier!r}")
            _validate_record(
                value,
                identifier,
                spec,
                location=f"submission[{index}]",
            )
            ordered_ids.append(identifier)
    else:
        if not isinstance(payload, Mapping):
            raise SubmissionError("Submission root must be a JSON object")
        ordered_ids = []
        for index, (raw_identifier, value) in enumerate(payload.items()):
            identifier = _canonical_id(
                raw_identifier,
                location=f"submission key[{index}]",
            )
            if identifier in ordered_ids:
                raise SubmissionError(f"Duplicate submission ID: {identifier!r}")
            if spec.dict_mode == "answer_map":
                _validate_answer(value, location=f"submission[{identifier!r}]")
            else:
                if not isinstance(value, Mapping):
                    raise SubmissionError(
                        f"submission[{identifier!r}]: record-map value must be "
                        "an object"
                    )
                expected_fields = set(spec.fields) - {spec.id_field}
                if set(value) != expected_fields:
                    raise SubmissionError(
                        f"submission[{identifier!r}]: fields do not match official "
                        f"record-map schema; expected={sorted(expected_fields)}, "
                        f"actual={sorted(value)}"
                    )
                _validate_answer(
                    value[spec.answer_field],
                    location=f"submission[{identifier!r}].{spec.answer_field}",
                )
            ordered_ids.append(identifier)

    expected_ids = _ordered_ids(
        parsed_targets,
        set(ordered_ids),
        spec.ordering,
    )
    if tuple(ordered_ids) != expected_ids:
        raise SubmissionError(
            "Submission IDs are not in the required deterministic order; "
            f"expected={list(expected_ids)}, actual={ordered_ids}"
        )
    return SubmissionValidation(len(ordered_ids), tuple(ordered_ids))


def load_submission(path: str | Path) -> Any:
    """Load an existing JSON submission with duplicate-key detection."""

    return _load_json(Path(path))


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SubmissionError(f"Refusing to overwrite existing submission: {path}")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise SubmissionError(f"Unable to atomically write submission: {path}") from exc
    finally:
        if temporary_path is not None and Path(temporary_path).exists():
            Path(temporary_path).unlink()


def create_submission_with_schema(
    predictions_path: str | Path,
    target_ids_path: str | Path,
    schema_path: str | Path,
    output_path: str | Path,
) -> SubmissionValidation:
    """Create an exact submission after validating every contract boundary."""

    spec = load_submission_spec(schema_path)
    predictions = load_predictions(predictions_path)
    target_ids = load_target_ids(target_ids_path)
    prediction_payload = [
        {"id": identifier, "answer": answer} for identifier, answer in predictions
    ]
    payload = build_submission(prediction_payload, target_ids, spec)
    validation = validate_submission(payload, target_ids=target_ids, spec=spec)
    _atomic_write_json(Path(output_path), payload)
    return validation


def validate_submission_file_with_schema(
    submission_path: str | Path,
    target_ids_path: str | Path,
    schema_path: str | Path,
) -> SubmissionValidation:
    """Validate an existing submission without rewriting it."""

    spec = load_submission_spec(schema_path)
    target_ids = load_target_ids(target_ids_path)
    payload = load_submission(submission_path)
    return validate_submission(payload, target_ids=target_ids, spec=spec)


# ---------------------------------------------------------------------------
# Fixed official submission contract
# ---------------------------------------------------------------------------

OFFICIAL_JSON_FILENAME = "submission.json"
OFFICIAL_ZIP_FILENAME = "submission.zip"

_OFFICIAL_PREDICTION_FIELDS = _PREDICTION_FIELDS | {
    "raw_answer",
    "cleaned_answer",
    # Reader artifacts keep model/provenance metadata alongside the official
    # ``id``/``answer`` pair. These fields are accepted only at the input
    # boundary and are never copied into submission.json.
    "model",
    "model_version",
    "confidence",
    "source_case_id",
    "source_origin",
}


@dataclass(frozen=True, slots=True)
class SubmissionIssue:
    """One typed diagnostic produced at an official submission boundary."""

    code: str
    message: str
    question_id: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionValidationReport:
    """Content and layout validation result for official JSON/ZIP artifacts."""

    valid: bool
    record_count: int
    expected_count: int
    ordered_ids: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()
    extra_ids: tuple[str, ...] = ()
    issues: tuple[SubmissionIssue, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Return unique error codes in first-observed order."""

        return tuple(dict.fromkeys(issue.code for issue in self.issues))

    @property
    def errors(self) -> tuple[SubmissionIssue, ...]:
        """Compatibility alias for callers that use an errors attribute."""

        return self.issues

    @property
    def question_count(self) -> int:
        """Return the number of unique question IDs observed in the artifact."""

        return self.record_count

    @property
    def invalid_records(self) -> tuple[SubmissionIssue, ...]:
        """Return per-record diagnostics under the report name used by the task."""

        return self.issues


class _OfficialJSONObject(list[tuple[str, Any]]):
    """Retain object pairs so duplicate IDs/fields cannot be overwritten."""


def _official_object_pairs_hook(
    pairs: list[tuple[str, Any]],
) -> _OfficialJSONObject:
    return _OfficialJSONObject(pairs)


def _official_issue(
    code: str,
    message: str,
    *,
    question_id: str | None = None,
) -> SubmissionIssue:
    return SubmissionIssue(code=code, message=message, question_id=question_id)


def _official_identifier(value: object, *, location: str) -> str:
    """Canonicalize only integer IDs; preserve string spelling exactly."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SubmissionError(
            f"{location}: question ID must be a string or integer",
            code="SUBMISSION_INVALID_QUESTION_ID",
        )
    if isinstance(value, str):
        if not value.strip():
            raise SubmissionError(
                f"{location}: question ID must not be blank",
                code="SUBMISSION_INVALID_QUESTION_ID",
            )
        return value
    return str(value)


def _official_identifier_sequence(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        identifier = _official_identifier(value, location=f"expected[{index}]")
        if identifier in seen:
            raise SubmissionError(
                f"Duplicate expected question ID: {identifier!r}",
                code="SUBMISSION_DUPLICATE_QUESTION_ID",
            )
        seen.add(identifier)
        result.append(identifier)
    if not result:
        raise SubmissionError(
            "Expected question ID set must not be empty",
            code="SUBMISSION_INVALID_QUESTION_ID",
        )
    return tuple(result)


def _official_prediction_mapping(
    value: object, *, location: str
) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    raise SubmissionError(
        f"{location}: prediction must be an object",
        code="SUBMISSION_VALUE_NOT_OBJECT",
    )


def _official_prediction_row(value: object, *, location: str) -> tuple[str, str]:
    mapping = _official_prediction_mapping(value, location=location)
    missing = {"id", "answer"} - set(mapping)
    if missing:
        raise SubmissionError(
            f"{location}: prediction is missing required field(s): "
            + ", ".join(sorted(missing)),
            code=(
                "SUBMISSION_INVALID_QUESTION_ID"
                if "id" in missing
                else "SUBMISSION_MISSING_ANSWER"
            ),
        )
    extra = set(mapping) - _OFFICIAL_PREDICTION_FIELDS
    if extra:
        raise SubmissionError(
            f"{location}: unsupported prediction field(s): {', '.join(sorted(extra))}",
            code="SUBMISSION_EXTRA_FIELDS",
        )
    identifier = _official_identifier(mapping["id"], location=f"{location}.id")
    answer = mapping["answer"]
    if not isinstance(answer, str):
        raise SubmissionError(
            f"{location}.answer: answer must be a string",
            code="SUBMISSION_ANSWER_NOT_STRING",
        )
    return identifier, answer


def build_submission_payload(
    predictions: Iterable[object],
    expected_question_ids: Iterable[object],
) -> dict[str, dict[str, str]]:
    """Build the fixed ``{question_id: {answer: string}}`` payload.

    The expected-ID iterable defines both coverage and output order.  Prediction
    metadata is accepted only as an internal input boundary and is never copied.
    """

    expected_ids = _official_identifier_sequence(expected_question_ids)
    expected_set = set(expected_ids)
    answers: dict[str, str] = {}
    for index, prediction in enumerate(predictions):
        identifier, answer = _official_prediction_row(
            prediction,
            location=f"prediction[{index}]",
        )
        if identifier in answers:
            raise SubmissionError(
                f"Duplicate prediction question ID: {identifier!r}",
                code="SUBMISSION_DUPLICATE_QUESTION_ID",
            )
        answers[identifier] = answer

    prediction_set = set(answers)
    missing = expected_set - prediction_set
    extra = prediction_set - expected_set
    if missing:
        raise SubmissionError(
            "Predictions are missing question ID(s): " + ", ".join(sorted(missing)),
            code="SUBMISSION_MISSING_QUESTION_ID",
        )
    if extra:
        raise SubmissionError(
            "Predictions contain extra question ID(s): " + ", ".join(sorted(extra)),
            code="SUBMISSION_EXTRA_QUESTION_ID",
        )
    return {identifier: {"answer": answers[identifier]} for identifier in expected_ids}


def _read_official_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise SubmissionError(
            f"File missing: {path}", code="SUBMISSION_FILE_NOT_FOUND"
        ) from exc
    except OSError as exc:
        raise SubmissionError(
            f"Unable to read file: {path}", code="SUBMISSION_FILE_NOT_FOUND"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(
            f"File must be UTF-8: {path}", code="SUBMISSION_INVALID_UTF8"
        ) from exc
    try:
        return json.loads(text, object_pairs_hook=_official_object_pairs_hook)
    except json.JSONDecodeError as exc:
        raise SubmissionError(
            f"Invalid JSON in {path}: {exc}", code="SUBMISSION_INVALID_JSON"
        ) from exc


def _read_official_jsonl(path: Path) -> list[Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise SubmissionError(
            f"File missing: {path}", code="SUBMISSION_FILE_NOT_FOUND"
        ) from exc
    except OSError as exc:
        raise SubmissionError(
            f"Unable to read file: {path}", code="SUBMISSION_FILE_NOT_FOUND"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(
            f"File must be UTF-8: {path}", code="SUBMISSION_INVALID_UTF8"
        ) from exc
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(
                json.loads(line, object_pairs_hook=_official_object_pairs_hook)
            )
        except json.JSONDecodeError as exc:
            raise SubmissionError(
                f"Invalid JSONL at {path}:{line_number}: {exc}",
                code="SUBMISSION_INVALID_JSON",
            ) from exc
    return records


def _pairs_to_field_map(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, _OfficialJSONObject):
        raise SubmissionError(
            f"{location}: value must be an object",
            code="SUBMISSION_VALUE_NOT_OBJECT",
        )
    fields: dict[str, object] = {}
    for field_name, field_value in value:
        if field_name in fields:
            raise SubmissionError(
                f"{location}: duplicate field {field_name!r}",
                code="SUBMISSION_EXTRA_FIELDS",
            )
        fields[field_name] = field_value
    return fields


def load_submission_question_ids(path: str | Path) -> tuple[str, ...]:
    """Load expected IDs in source dataset order without consuming answers.

    Supported inputs are the observed ID-keyed question map, a list of records
    with ``id``, an ``ids``/``records`` wrapper, or JSONL records with ``id``.
    Values under question-map keys are intentionally not inspected.
    """

    source_path = Path(path)
    payload: Any
    if source_path.suffix.casefold() in {".jsonl", ".ndjson"}:
        payload = _read_official_jsonl(source_path)
    else:
        payload = _read_official_json(source_path)

    raw_ids: list[object]
    if isinstance(payload, _OfficialJSONObject):
        pairs = list(payload)
        keys = [key for key, _ in pairs]
        if keys == ["ids"]:
            if not isinstance(pairs[0][1], list):
                raise SubmissionError(
                    f"{source_path}: ids must be a list",
                    code="SUBMISSION_INVALID_QUESTION_ID",
                )
            raw_ids = list(pairs[0][1])
        elif keys == ["records"]:
            if not isinstance(pairs[0][1], list):
                raise SubmissionError(
                    f"{source_path}: records must be a list",
                    code="SUBMISSION_INVALID_QUESTION_ID",
                )
            raw_ids = []
            for index, record in enumerate(pairs[0][1]):
                fields = _pairs_to_field_map(record, location=f"record[{index}]")
                if "id" not in fields:
                    raise SubmissionError(
                        f"record[{index}]: missing id",
                        code="SUBMISSION_INVALID_QUESTION_ID",
                    )
                raw_ids.append(fields["id"])
        else:
            # The normal question dataset is an ID-keyed map.  Only its keys
            # are used, so an answer field can never enter this inference path.
            raw_ids = []
            raw_ids.extend(keys)
    elif isinstance(payload, list):
        raw_ids = []
        for index, record in enumerate(payload):
            if isinstance(record, _OfficialJSONObject):
                fields = _pairs_to_field_map(record, location=f"record[{index}]")
                if "id" not in fields:
                    raise SubmissionError(
                        f"record[{index}]: missing id",
                        code="SUBMISSION_INVALID_QUESTION_ID",
                    )
                raw_ids.append(fields["id"])
            else:
                raw_ids.append(record)
    else:
        raise SubmissionError(
            f"{source_path}: question dataset must be an object or list",
            code="SUBMISSION_TOP_LEVEL_NOT_OBJECT",
        )
    return _official_identifier_sequence(raw_ids)


def _report_from_json_bytes(
    raw: bytes,
    *,
    source_name: str,
    expected_question_ids: Iterable[object],
    reject_empty_answers: bool,
) -> SubmissionValidationReport:
    issues: list[SubmissionIssue] = []
    warnings: list[str] = []
    try:
        expected_ids = _official_identifier_sequence(expected_question_ids)
    except SubmissionError as exc:
        expected_ids = ()
        issues.append(_official_issue(exc.code, str(exc)))

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=len(expected_ids),
            issues=tuple(
                [*issues, _official_issue("SUBMISSION_INVALID_UTF8", source_name)]
            ),
        )
    try:
        payload = json.loads(text, object_pairs_hook=_official_object_pairs_hook)
    except json.JSONDecodeError as exc:
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=len(expected_ids),
            issues=tuple(
                [
                    *issues,
                    _official_issue("SUBMISSION_INVALID_JSON", f"{source_name}: {exc}"),
                ]
            ),
        )

    if not isinstance(payload, _OfficialJSONObject):
        issues.append(
            _official_issue(
                "SUBMISSION_TOP_LEVEL_NOT_OBJECT",
                f"{source_name}: top level must be a JSON object",
            )
        )
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=len(expected_ids),
            issues=tuple(issues),
        )

    ordered_ids: list[str] = []
    values: dict[str, object] = {}
    seen: set[str] = set()
    for raw_identifier, value in payload:
        try:
            identifier = _official_identifier(
                raw_identifier,
                location=f"{source_name} question ID",
            )
        except SubmissionError as exc:
            issues.append(_official_issue(exc.code, str(exc)))
            continue
        if identifier in seen:
            issues.append(
                _official_issue(
                    "SUBMISSION_DUPLICATE_QUESTION_ID",
                    f"{source_name}: duplicate question ID {identifier!r}",
                    question_id=identifier,
                )
            )
            continue
        seen.add(identifier)
        ordered_ids.append(identifier)
        values[identifier] = value

    expected_set = set(expected_ids)
    actual_set = set(ordered_ids)
    missing = tuple(
        identifier for identifier in expected_ids if identifier not in actual_set
    )
    extra = tuple(
        identifier for identifier in ordered_ids if identifier not in expected_set
    )
    if missing:
        issues.append(
            _official_issue(
                "SUBMISSION_MISSING_QUESTION_ID",
                "Missing question ID(s): " + ", ".join(missing),
            )
        )
    if extra:
        issues.append(
            _official_issue(
                "SUBMISSION_EXTRA_QUESTION_ID",
                "Extra question ID(s): " + ", ".join(extra),
            )
        )

    for identifier in ordered_ids:
        value = values[identifier]
        location = f"{source_name}[{identifier!r}]"
        if not isinstance(value, _OfficialJSONObject):
            issues.append(
                _official_issue(
                    "SUBMISSION_VALUE_NOT_OBJECT",
                    f"{location}: value must be an object",
                    question_id=identifier,
                )
            )
            continue
        try:
            fields = _pairs_to_field_map(value, location=location)
        except SubmissionError as exc:
            issues.append(_official_issue(exc.code, str(exc), question_id=identifier))
            continue
        if "answer" not in fields:
            issues.append(
                _official_issue(
                    "SUBMISSION_MISSING_ANSWER",
                    f"{location}: missing answer field",
                    question_id=identifier,
                )
            )
        extra_fields = set(fields) - {"answer"}
        if extra_fields:
            issues.append(
                _official_issue(
                    "SUBMISSION_EXTRA_FIELDS",
                    f"{location}: extra field(s): {', '.join(sorted(extra_fields))}",
                    question_id=identifier,
                )
            )
        if "answer" in fields:
            answer = fields["answer"]
            if not isinstance(answer, str):
                issues.append(
                    _official_issue(
                        "SUBMISSION_ANSWER_NOT_STRING",
                        f"{location}.answer: answer must be a string",
                        question_id=identifier,
                    )
                )
            elif answer == "":
                if reject_empty_answers:
                    issues.append(
                        _official_issue(
                            "SUBMISSION_EMPTY_ANSWER",
                            f"{location}.answer: answer must not be empty",
                            question_id=identifier,
                        )
                    )
                else:
                    warnings.append(f"EMPTY_ANSWER:{identifier}")

    if tuple(ordered_ids) != expected_ids and not missing and not extra:
        issues.append(
            _official_issue(
                "SUBMISSION_INVALID_QUESTION_ID",
                f"{source_name}: question IDs are not in dataset order",
            )
        )
    return SubmissionValidationReport(
        valid=not issues,
        record_count=len(ordered_ids),
        expected_count=len(expected_ids),
        ordered_ids=tuple(ordered_ids),
        missing_ids=missing,
        extra_ids=extra,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _raise_for_report(report: SubmissionValidationReport) -> None:
    if report.valid:
        return
    issue = (
        report.issues[0]
        if report.issues
        else _official_issue("SUBMISSION_INVALID_JSON", "Submission validation failed")
    )
    raise SubmissionError(issue.message, code=issue.code)


def validate_submission_json(
    submission_path: str | Path,
    expected_question_ids: Iterable[object],
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Validate the fixed official JSON file without rewriting it."""

    path = Path(submission_path)
    issues: list[SubmissionIssue] = []
    if path.name != OFFICIAL_JSON_FILENAME:
        issues.append(
            _official_issue(
                "SUBMISSION_WRONG_FILENAME",
                f"JSON submission must be named {OFFICIAL_JSON_FILENAME!r}",
            )
        )
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=0,
            issues=tuple(
                [
                    *issues,
                    _official_issue(
                        "SUBMISSION_FILE_NOT_FOUND", f"File missing: {path}"
                    ),
                ]
            ),
        )
    except OSError as exc:
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=0,
            issues=tuple(
                [
                    *issues,
                    _official_issue(
                        "SUBMISSION_FILE_NOT_FOUND", f"Unable to read {path}: {exc}"
                    ),
                ]
            ),
        )
    report = _report_from_json_bytes(
        raw,
        source_name=str(path),
        expected_question_ids=tuple(expected_question_ids),
        reject_empty_answers=reject_empty_answers,
    )
    combined_issues = tuple([*issues, *report.issues])
    return SubmissionValidationReport(
        valid=not combined_issues,
        record_count=report.record_count,
        expected_count=report.expected_count,
        ordered_ids=report.ordered_ids,
        missing_ids=report.missing_ids,
        extra_ids=report.extra_ids,
        issues=tuple([*issues, *report.issues]),
        warnings=report.warnings,
    )


def _validate_zip_archive(
    path: Path,
    expected_question_ids: Iterable[object],
    *,
    enforce_filename: bool,
    reject_empty_answers: bool,
) -> SubmissionValidationReport:
    issues: list[SubmissionIssue] = []
    if enforce_filename and path.name != OFFICIAL_ZIP_FILENAME:
        issues.append(
            _official_issue(
                "SUBMISSION_WRONG_FILENAME",
                f"ZIP submission must be named {OFFICIAL_ZIP_FILENAME!r}",
            )
        )
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if not members:
                issues.append(
                    _official_issue(
                        "SUBMISSION_ZIP_MISSING_JSON",
                        "ZIP does not contain submission.json",
                    )
                )
            if names != [OFFICIAL_JSON_FILENAME]:
                issues.append(
                    _official_issue(
                        "SUBMISSION_ZIP_WRONG_LAYOUT",
                        f"ZIP members must be exactly [{OFFICIAL_JSON_FILENAME!r}]",
                    )
                )
            if not members:
                return SubmissionValidationReport(
                    valid=False,
                    record_count=0,
                    expected_count=0,
                    issues=tuple(issues),
                )
            if len(members) != 1:
                issues.append(
                    _official_issue(
                        "SUBMISSION_ZIP_EXTRA_MEMBER",
                        "ZIP must contain exactly one member named submission.json",
                    )
                )
            if OFFICIAL_JSON_FILENAME not in names:
                issues.append(
                    _official_issue(
                        "SUBMISSION_ZIP_MISSING_JSON",
                        "ZIP is missing root member submission.json",
                    )
                )
                return SubmissionValidationReport(
                    valid=False,
                    record_count=0,
                    expected_count=0,
                    issues=tuple(issues),
                )
            member = next(
                member
                for member in members
                if member.filename == OFFICIAL_JSON_FILENAME
            )
            try:
                raw_json = archive.read(member)
            except (BadZipFile, OSError, RuntimeError) as exc:
                issues.append(
                    _official_issue(
                        "SUBMISSION_ZIP_INVALID",
                        f"Unable to read ZIP member {OFFICIAL_JSON_FILENAME}: {exc}",
                    )
                )
                return SubmissionValidationReport(
                    valid=False,
                    record_count=0,
                    expected_count=0,
                    issues=tuple(issues),
                )
    except (BadZipFile, OSError, RuntimeError) as exc:
        issues.append(
            _official_issue("SUBMISSION_ZIP_INVALID", f"Invalid ZIP: {path}: {exc}")
        )
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=0,
            issues=tuple(issues),
        )

    report = _report_from_json_bytes(
        raw_json,
        source_name=OFFICIAL_JSON_FILENAME,
        expected_question_ids=tuple(expected_question_ids),
        reject_empty_answers=reject_empty_answers,
    )
    combined_issues = tuple([*issues, *report.issues])
    return SubmissionValidationReport(
        valid=not combined_issues,
        record_count=report.record_count,
        expected_count=report.expected_count,
        ordered_ids=report.ordered_ids,
        missing_ids=report.missing_ids,
        extra_ids=report.extra_ids,
        issues=tuple([*issues, *report.issues]),
        warnings=report.warnings,
    )


def validate_submission_zip(
    submission_zip_path: str | Path,
    expected_question_ids: Iterable[object],
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Validate final ZIP filename, exact layout, encoding, schema, and coverage."""

    path = Path(submission_zip_path)
    if not path.exists():
        return SubmissionValidationReport(
            valid=False,
            record_count=0,
            expected_count=0,
            issues=(
                _official_issue("SUBMISSION_FILE_NOT_FOUND", f"File missing: {path}"),
            ),
        )
    return _validate_zip_archive(
        path,
        expected_question_ids,
        enforce_filename=True,
        reject_empty_answers=reject_empty_answers,
    )


def write_submission_json(
    payload: object,
    output_path: str | Path,
    expected_question_ids: Iterable[object],
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Write deterministic UTF-8 JSON and validate the bytes after writing."""

    path = Path(output_path)
    expected_ids = tuple(expected_question_ids)
    if path.name != OFFICIAL_JSON_FILENAME:
        raise SubmissionError(
            f"JSON submission must be named {OFFICIAL_JSON_FILENAME!r}",
            code="SUBMISSION_WRONG_FILENAME",
        )
    if path.exists():
        raise SubmissionError(
            f"Refusing to overwrite existing submission: {path}",
            code="SUBMISSION_INVALID_JSON",
        )
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise SubmissionError(
            f"Submission payload is not JSON serializable: {exc}",
            code="SUBMISSION_INVALID_JSON",
        ) from exc
    raw = (serialized + "\n").encode("utf-8")
    report = _report_from_json_bytes(
        raw,
        source_name=str(path),
        expected_question_ids=expected_ids,
        reject_empty_answers=reject_empty_answers,
    )
    _raise_for_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(raw)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise SubmissionError(
            f"Unable to atomically write submission: {path}",
            code="SUBMISSION_INVALID_JSON",
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    parsed = validate_submission_json(
        path,
        expected_ids,
        reject_empty_answers=reject_empty_answers,
    )
    _raise_for_report(parsed)
    return parsed


def create_submission_zip(
    submission_json_path: str | Path,
    output_zip_path: str | Path,
    expected_question_ids: Iterable[object],
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Package one validated JSON member as deterministic ``submission.zip``."""

    json_path = Path(submission_json_path)
    expected_ids = tuple(expected_question_ids)
    zip_path = Path(output_zip_path)
    if zip_path.name != OFFICIAL_ZIP_FILENAME:
        raise SubmissionError(
            f"ZIP submission must be named {OFFICIAL_ZIP_FILENAME!r}",
            code="SUBMISSION_WRONG_FILENAME",
        )
    if zip_path.exists():
        raise SubmissionError(
            f"Refusing to overwrite existing submission ZIP: {zip_path}",
            code="SUBMISSION_ZIP_INVALID",
        )
    json_report = validate_submission_json(
        json_path,
        expected_ids,
        reject_empty_answers=reject_empty_answers,
    )
    _raise_for_report(json_report)
    raw_json = json_path.read_bytes()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=zip_path.parent,
            prefix=f".{zip_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        info = ZipInfo(OFFICIAL_JSON_FILENAME)
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        with ZipFile(temporary_path, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(info, raw_json)
        temporary_report = _validate_zip_archive(
            temporary_path,
            expected_ids,
            enforce_filename=False,
            reject_empty_answers=reject_empty_answers,
        )
        _raise_for_report(temporary_report)
        os.replace(temporary_path, zip_path)
    except (OSError, BadZipFile, RuntimeError) as exc:
        raise SubmissionError(
            f"Unable to create valid submission ZIP: {zip_path}",
            code="SUBMISSION_ZIP_INVALID",
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    result = validate_submission_zip(
        zip_path,
        expected_ids,
        reject_empty_answers=reject_empty_answers,
    )
    _raise_for_report(result)
    return result


def create_submission(
    predictions_path: str | Path,
    questions_path: str | Path,
    output_zip_path: str | Path,
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Create the final official ZIP from predictions and inference IDs."""

    predictions = load_predictions(predictions_path)
    expected_ids = load_submission_question_ids(questions_path)
    payload = build_submission_payload(
        ({"id": identifier, "answer": answer} for identifier, answer in predictions),
        expected_ids,
    )
    with tempfile.TemporaryDirectory(prefix="legal-rag-submission-") as temporary:
        json_path = Path(temporary) / OFFICIAL_JSON_FILENAME
        write_submission_json(
            payload,
            json_path,
            expected_ids,
            reject_empty_answers=reject_empty_answers,
        )
        return create_submission_zip(
            json_path,
            output_zip_path,
            expected_ids,
            reject_empty_answers=reject_empty_answers,
        )


def validate_submission_file(
    submission_zip_path: str | Path,
    questions_path: str | Path,
    *,
    reject_empty_answers: bool = False,
) -> SubmissionValidationReport:
    """Validate the final official ZIP against IDs from the question dataset."""

    expected_ids = load_submission_question_ids(questions_path)
    return validate_submission_zip(
        submission_zip_path,
        expected_ids,
        reject_empty_answers=reject_empty_answers,
    )
