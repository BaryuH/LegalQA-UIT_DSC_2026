"""VAL-00: leakage-safe clean warmup validation manifest (IDs-only)."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from legal_rag.finetuned_reader.split_remediation import normalize_question_text
from legal_rag.questions import (
    QuestionLoadError,
    inspect_questions,
    load_questions,
)

POLICY_ID = "sedar-warmup-public-exclusion-v1"
NORMALIZATION_POLICY_ID = "ftr03-train-overlap-exclusion-v1"
NORMALIZATION_IMPLEMENTATION = (
    "legal_rag.finetuned_reader.split_remediation.normalize_question_text"
)
SCHEMA_VERSION = 1

ExclusionReason = Literal[
    "PUBLIC_ID_OVERLAP",
    "PUBLIC_NORMALIZED_QUESTION_OVERLAP",
    "PUBLIC_ID_AND_QUESTION_OVERLAP",
]


class WarmupValidationError(ValueError):
    """Raised when the clean warmup validation view cannot be built safely."""


@dataclass(frozen=True, slots=True)
class WarmupPublicExclusion:
    """One warmup case excluded from the local validation view."""

    case_id: str
    reason: ExclusionReason

    def as_dict(self) -> dict[str, str]:
        return {"case_id": self.case_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class CleanWarmupValidationResult:
    """Deterministic IDs-only validation artifacts and provenance."""

    manifest: dict[str, Any]
    overlap_report: dict[str, Any]
    audit: dict[str, Any]

    @property
    def included_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["included_ids"])

    @property
    def included_ids_hash(self) -> str:
        return str(self.manifest["included_ids_hash"])

    @property
    def excluded_ids_hash(self) -> str:
        return str(self.manifest["excluded_ids_hash"])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _ids_hash(ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(ids))


def normalization_implementation_hash() -> str:
    """Stable hash of the approved FTR-03 normalizer source."""

    return _sha256_text(inspect.getsource(normalize_question_text))


def _write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            temporary.unlink(missing_ok=True)
            return
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write sorted JSON with trailing newline via atomic replace."""

    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_atomic_text(path, content)


def derive_warmup_public_exclusions(
    warmup_questions_by_id: Mapping[str, str],
    public_questions_by_id: Mapping[str, str],
) -> tuple[WarmupPublicExclusion, ...]:
    """Exclude warmup cases overlapping Public by ID or normalized question.

    Selection uses only IDs and the approved FTR-03 ``normalize_question_text``.
    Answer text is never consulted.
    """

    public_ids = set(public_questions_by_id)
    public_normalized = {
        normalize_question_text(question)
        for question in public_questions_by_id.values()
    }
    exclusions: list[WarmupPublicExclusion] = []
    for case_id in sorted(warmup_questions_by_id):
        question = warmup_questions_by_id[case_id]
        id_overlap = case_id in public_ids
        question_overlap = normalize_question_text(question) in public_normalized
        if id_overlap and question_overlap:
            reason: ExclusionReason = "PUBLIC_ID_AND_QUESTION_OVERLAP"
        elif id_overlap:
            reason = "PUBLIC_ID_OVERLAP"
        elif question_overlap:
            reason = "PUBLIC_NORMALIZED_QUESTION_OVERLAP"
        else:
            continue
        exclusions.append(WarmupPublicExclusion(case_id=case_id, reason=reason))
    return tuple(exclusions)


def _load_question_map(path: Path, *, split: str) -> dict[str, str]:
    """Load ID→question without materializing answers."""

    try:
        records = load_questions(path, split=split, include_answers=False)
    except QuestionLoadError as exc:
        raise WarmupValidationError(str(exc)) from exc
    return {record.id: record.question for record in records}


def _fail_closed_warmup_gates(path: Path) -> dict[str, Any]:
    """Fail closed on duplicate IDs or blank warmup questions; record diagnostics."""

    try:
        stats = inspect_questions(path)
    except QuestionLoadError as exc:
        raise WarmupValidationError(str(exc)) from exc

    if stats.duplicate_id_count:
        raise WarmupValidationError(
            f"{path}: duplicate question IDs detected "
            f"(count={stats.duplicate_id_count}); fail closed"
        )
    if stats.blank_question_count:
        raise WarmupValidationError(
            f"{path}: blank warmup question(s) detected "
            f"(count={stats.blank_question_count}); fail closed"
        )
    return {
        "record_count": stats.record_count,
        "duplicate_id_count": stats.duplicate_id_count,
        "blank_question_count": stats.blank_question_count,
        "blank_answer_count": stats.blank_answer_count,
        "answer_missing_count": stats.answer_missing_count,
    }


def build_clean_warmup_validation(
    *,
    warmup_path: Path,
    public_path: Path,
) -> CleanWarmupValidationResult:
    """Build the IDs-only clean warmup validation view and provenance reports."""

    warmup_path = Path(warmup_path)
    public_path = Path(public_path)
    if not warmup_path.is_file():
        raise WarmupValidationError(f"Warmup source missing: {warmup_path}")
    if not public_path.is_file():
        raise WarmupValidationError(f"Public source missing: {public_path}")

    warmup_stats = _fail_closed_warmup_gates(warmup_path)
    source_warmup_sha256 = _sha256_file(warmup_path)
    source_public_sha256 = _sha256_file(public_path)

    warmup_questions = _load_question_map(warmup_path, split="warmup")
    public_questions = _load_question_map(public_path, split="public")

    blank_loaded = sum(
        1 for question in warmup_questions.values() if not question.strip()
    )
    if blank_loaded:
        raise WarmupValidationError(
            f"{warmup_path}: blank warmup question(s) after load "
            f"(count={blank_loaded}); fail closed"
        )

    exclusions = derive_warmup_public_exclusions(warmup_questions, public_questions)
    excluded_ids = tuple(item.case_id for item in exclusions)
    excluded_id_set = set(excluded_ids)
    included_ids = tuple(
        case_id
        for case_id in sorted(warmup_questions)
        if case_id not in excluded_id_set
    )

    public_ids = set(public_questions)
    public_normalized = {
        normalize_question_text(question) for question in public_questions.values()
    }
    exact_id_overlap_ids = sorted(set(warmup_questions) & public_ids)
    normalized_overlap_ids = sorted(
        case_id
        for case_id, question in warmup_questions.items()
        if normalize_question_text(question) in public_normalized
    )

    norm_hash = normalization_implementation_hash()
    included_hash = _ids_hash(included_ids)
    excluded_hash = _ids_hash(excluded_ids)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "source_warmup_sha256": source_warmup_sha256,
        "source_public_sha256": source_public_sha256,
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "normalization_implementation": NORMALIZATION_IMPLEMENTATION,
        "normalization_hash_or_version": norm_hash,
        "source_warmup_count": len(warmup_questions),
        "excluded_count": len(excluded_ids),
        "included_count": len(included_ids),
        "excluded_ids_hash": excluded_hash,
        "included_ids_hash": included_hash,
        "included_ids": list(included_ids),
    }

    reason_counts: dict[str, int] = {
        "PUBLIC_ID_OVERLAP": 0,
        "PUBLIC_NORMALIZED_QUESTION_OVERLAP": 0,
        "PUBLIC_ID_AND_QUESTION_OVERLAP": 0,
    }
    for item in exclusions:
        reason_counts[item.reason] += 1

    overlap_report = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "source_warmup_count": len(warmup_questions),
        "source_public_count": len(public_questions),
        "exact_id_overlap_count": len(exact_id_overlap_ids),
        "normalized_question_overlap_count": len(normalized_overlap_ids),
        "union_excluded_count": len(excluded_ids),
        "included_count": len(included_ids),
        "reason_counts": reason_counts,
        "exact_id_overlap_ids": exact_id_overlap_ids,
        "normalized_question_overlap_ids": normalized_overlap_ids,
        "excluded": [item.as_dict() for item in exclusions],
    }

    audit = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "normalization_implementation": NORMALIZATION_IMPLEMENTATION,
        "normalization_hash_or_version": norm_hash,
        "source_warmup_path": warmup_path.as_posix(),
        "source_public_path": public_path.as_posix(),
        "source_warmup_sha256_before": source_warmup_sha256,
        "source_public_sha256_before": source_public_sha256,
        "source_warmup_sha256_after": _sha256_file(warmup_path),
        "source_public_sha256_after": _sha256_file(public_path),
        "source_hashes_unchanged": True,
        "warmup_inspect": warmup_stats,
        "blank_warmup_references": warmup_stats["blank_answer_count"],
        "duplicate_ids": warmup_stats["duplicate_id_count"],
        "blank_questions": warmup_stats["blank_question_count"],
        "source_warmup_count": len(warmup_questions),
        "source_public_count": len(public_questions),
        "exact_id_overlap_count": len(exact_id_overlap_ids),
        "normalized_question_overlap_count": len(normalized_overlap_ids),
        "union_excluded_count": len(excluded_ids),
        "included_count": len(included_ids),
        "included_ids_hash": included_hash,
        "excluded_ids_hash": excluded_hash,
        "determinism_check": {
            "included_ids_sorted": list(included_ids) == sorted(included_ids),
            "excluded_ids_sorted": list(excluded_ids) == sorted(excluded_ids),
            "repeat_build_included_ids_hash": included_hash,
        },
        "answers_used_for_selection": False,
        "public_answers_materialized": False,
        "warmup_answers_materialized": False,
    }

    if (
        audit["source_warmup_sha256_before"] != audit["source_warmup_sha256_after"]
        or audit["source_public_sha256_before"] != audit["source_public_sha256_after"]
    ):
        audit["source_hashes_unchanged"] = False
        raise WarmupValidationError(
            "Source data hashes changed during validation build; refuse to continue"
        )

    return CleanWarmupValidationResult(
        manifest=manifest,
        overlap_report=overlap_report,
        audit=audit,
    )


def write_clean_warmup_validation_artifacts(
    result: CleanWarmupValidationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the three VAL-00 JSON artifacts under ``output_dir``."""

    output_dir = Path(output_dir)
    paths = {
        "manifest": output_dir / "clean_warmup_manifest.json",
        "overlap_report": output_dir / "warmup_public_overlap_report.json",
        "audit": output_dir / "warmup_validation_audit.json",
    }
    write_json_atomic(paths["manifest"], result.manifest)
    write_json_atomic(paths["overlap_report"], result.overlap_report)
    write_json_atomic(paths["audit"], result.audit)
    return paths


__all__ = [
    "CleanWarmupValidationResult",
    "ExclusionReason",
    "NORMALIZATION_IMPLEMENTATION",
    "NORMALIZATION_POLICY_ID",
    "POLICY_ID",
    "SCHEMA_VERSION",
    "WarmupPublicExclusion",
    "WarmupValidationError",
    "build_clean_warmup_validation",
    "derive_warmup_public_exclusions",
    "normalization_implementation_hash",
    "normalize_question_text",
    "write_clean_warmup_validation_artifacts",
    "write_json_atomic",
]
