"""Reusable read-only validation and content-free reporting for source data."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.verify_data_manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestVerificationError,
    build_manifest,
    verify_manifest,
)

from .config import ProjectConfig
from .contexts import ContextLoadError, load_selected_contexts
from .questions import QuestionLoadError, inspect_questions, load_questions

REPORT_SCHEMA_VERSION = "b6.data-validation.v1"


@dataclass(frozen=True)
class ValidationRun:
    """Validation result and deterministic report-artifact destination."""

    report: dict[str, Any]
    output_path: Path

    @property
    def is_valid(self) -> bool:
        """Whether validation found no critical source or contract errors."""

        return not self.report["critical_errors"]


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _length_summary(lengths: Sequence[int]) -> dict[str, float | int | None]:
    """Return deterministic min/mean/p95/max character-length summary."""

    if not lengths:
        return {"min": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(lengths)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 6),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _archive_stats(path: Path, repo_root: Path) -> tuple[dict[str, Any], str | None]:
    """Inspect ZIP metadata only; never extract or print member content."""

    result: dict[str, Any] = {
        "path": _relative_path(path, repo_root),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "byte_size": path.stat().st_size if path.is_file() else None,
        "member_count": 0,
        "json_member_count": 0,
        "compressed_bytes": 0,
        "uncompressed_bytes": 0,
    }
    if not path.exists():
        return result, None
    if not path.is_file():
        return (
            result,
            "Selected-context archive is not a file: "
            f"{_relative_path(path, repo_root)}",
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
    except zipfile.BadZipFile:
        return (
            result,
            f"Invalid selected-context ZIP archive: {_relative_path(path, repo_root)}",
        )

    result.update(
        {
            "member_count": len(members),
            "json_member_count": sum(
                member.filename.casefold().endswith(".json") for member in members
            ),
            "compressed_bytes": sum(member.compress_size for member in members),
            "uncompressed_bytes": sum(member.file_size for member in members),
        }
    )
    return result, None


def _report_output_path(
    config: ProjectConfig, repo_root: Path, source_fingerprint: str
) -> Path:
    profile = config.project.profile.replace("/", "-").replace("\\", "-")
    filename = f"{profile}-{config.config_hash()[:12]}-{source_fingerprint[:12]}.json"
    return repo_root / config.runtime.artifacts_dir / "data-validation" / filename


def validate_data(config: ProjectConfig, repo_root: str | Path) -> ValidationRun:
    """Validate configured source inputs and return a JSON-serializable report.

    The function only reads source files. It uses the production question/context
    loaders for critical validation and reports aggregate lengths/counts only.
    """

    root = Path(repo_root).resolve()
    critical_errors: list[str] = []

    try:
        manifest_payload = build_manifest(root)
        source_fingerprint = _sha256_json(manifest_payload)
    except (ManifestVerificationError, OSError, ValueError) as exc:
        manifest_payload = {"files": []}
        source_fingerprint = "unavailable"
        critical_errors.append(f"Unable to inspect source files: {exc}")

    manifest_path = root / DEFAULT_MANIFEST_PATH
    try:
        verified_file_count = verify_manifest(root, manifest_path)
        manifest_status = {"status": "verified", "file_count": verified_file_count}
    except (ManifestVerificationError, OSError, ValueError) as exc:
        manifest_status = {"status": "failed", "file_count": None}
        critical_errors.append(f"Data manifest verification failed: {exc}")

    question_path = root / config.data.question_path
    question_report: dict[str, Any]
    try:
        diagnostics = inspect_questions(question_path)
        question_report = {
            "path": _relative_path(question_path, root),
            "count": diagnostics.record_count,
            "answer_availability": {
                "available": diagnostics.answer_available_count,
                "missing": diagnostics.answer_missing_count,
            },
            "duplicate_id_count": diagnostics.duplicate_id_count,
            "blank_counts": {
                "question": diagnostics.blank_question_count,
                "answer": diagnostics.blank_answer_count,
            },
            "lengths": {
                "question": _length_summary(diagnostics.question_lengths),
                "answer": _length_summary(diagnostics.answer_lengths),
            },
        }
    except QuestionLoadError as exc:
        question_report = {
            "path": _relative_path(question_path, root),
            "count": 0,
            "answer_availability": {"available": 0, "missing": 0},
            "duplicate_id_count": 0,
            "blank_counts": {"question": 0, "answer": 0},
            "lengths": {
                "question": _length_summary(()),
                "answer": _length_summary(()),
            },
        }
        critical_errors.append(f"Question inspection failed: {exc}")

    try:
        records = load_questions(question_path, split=config.data.split)
        question_report["validated_count"] = len(records)
    except QuestionLoadError as exc:
        question_report["validated_count"] = 0
        critical_errors.append(f"Question validation failed: {exc}")

    archive_path = root / config.data.selected_contexts_path
    archive_report, archive_error = _archive_stats(archive_path, root)
    context_count = 0
    context_error: str | None = archive_error
    if archive_error is None and archive_path.exists():
        try:
            context_count = len(load_selected_contexts(archive_path))
        except ContextLoadError as exc:
            context_error = str(exc)

    if context_error is not None:
        critical_errors.append(f"Context validation failed: {context_error}")
    elif not archive_path.exists() and config.retrieval.strategy != "none":
        critical_errors.append(
            "Context validation failed: retrieval requires selected-context archive "
            f"{_relative_path(archive_path, root)}"
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile": config.project.profile,
        "split": config.data.split,
        "split_policy": config.data.split_policy,
        "split_usage": config.split_usage.as_dict(),
        "config_hash": config.config_hash(),
        "manifest": {
            **manifest_status,
            "current_source_fingerprint": source_fingerprint,
            "files": manifest_payload.get("files", []),
        },
        "questions": question_report,
        "contexts": {"count": context_count, "archive": archive_report},
        "critical_errors": critical_errors,
    }
    return ValidationRun(
        report=report,
        output_path=_report_output_path(config, root, source_fingerprint),
    )


def write_validation_report(run: ValidationRun) -> None:
    """Atomically write a deterministic report without replacing differing output."""

    serialized = (
        json.dumps(
            run.report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path = run.output_path
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(
            f"Validation report already exists with different content: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(serialized, encoding="utf-8", newline="\n")
    temporary_path.replace(path)
