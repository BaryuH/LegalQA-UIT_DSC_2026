"""Read-only data loaders for questions and legal contexts.

Reads data in-place from ``data/`` without extracting or modifying source files.
Gold answers are isolated to the evaluation-only loader.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceQuestion:
    """Question-only view — no gold answer field."""

    id: str
    question: str
    split: str | None = None


@dataclass(frozen=True, slots=True)
class LegalDocument:
    """One legal context document with source provenance."""

    id: str
    name: str
    passage: str
    source_path: str
    content_hash: str
    link: str | None = None
    source_member: str | None = None


class DataLoadError(ValueError):
    """Raised when a data source cannot be loaded."""


def _read_question_map(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read the top-level JSON object preserving key order and detecting dupes."""
    if not path.is_file():
        raise DataLoadError(f"Question file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise DataLoadError(
            f"{path}: expected a JSON object, got {type(raw).__name__}"
        )

    seen: set[str] = set()
    pairs: list[tuple[str, dict[str, Any]]] = []
    for key, value in raw.items():
        str_key = str(key)
        if str_key in seen:
            raise DataLoadError(f"{path}: duplicate question ID {str_key!r}")
        seen.add(str_key)
        if not isinstance(value, dict):
            raise DataLoadError(
                f"{path}: record {str_key!r} must be a JSON object"
            )
        pairs.append((str_key, value))
    return pairs


def load_questions(path: str | Path, *, split: str | None = None) -> list[InferenceQuestion]:
    """Load inference-safe question views (no gold answers)."""
    pairs = _read_question_map(Path(path))
    questions: list[InferenceQuestion] = []
    for qid, record in pairs:
        question_text = record.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            raise DataLoadError(
                f"{path}: record {qid!r} has missing/blank question"
            )
        questions.append(
            InferenceQuestion(id=qid, question=question_text, split=split)
        )
    return questions


def load_gold_answers(path: str | Path) -> dict[str, str]:
    """Load gold answers for evaluation only. NEVER use in inference paths."""
    pairs = _read_question_map(Path(path))
    answers: dict[str, str] = {}
    for qid, record in pairs:
        answer_text = record.get("answer")
        if isinstance(answer_text, str) and answer_text.strip():
            answers[qid] = answer_text
    return answers


def load_legal_contexts(zip_path: str | Path) -> list[LegalDocument]:
    """Load legal contexts from selected-contexts.zip in-memory (no extraction).

    Reads each ``context_*.json`` member from the ZIP archive without
    extracting to disk, preserving ``data/`` immutability.
    """
    archive_path = Path(zip_path)
    if not archive_path.is_file():
        raise DataLoadError(f"Context archive not found: {archive_path}")

    documents: list[LegalDocument] = []
    warnings: list[str] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(
            name for name in zf.namelist()
            if name.endswith(".json") and "/" not in name
        )
        if not members:
            raise DataLoadError(
                f"No context_*.json members found in {archive_path}"
            )

        seen_ids: set[str] = set()
        for member_name in members:
            raw_bytes = zf.read(member_name)
            content_hash = sha256(raw_bytes).hexdigest()[:16]
            try:
                record = json.loads(raw_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                warnings.append(f"SKIP {member_name}: {exc}")
                continue

            if not isinstance(record, dict):
                warnings.append(f"SKIP {member_name}: not a JSON object")
                continue

            doc_id = record.get("id")
            if doc_id is None:
                warnings.append(f"SKIP {member_name}: missing 'id'")
                continue
            str_id = str(doc_id)

            if str_id in seen_ids:
                warnings.append(f"SKIP {member_name}: duplicate ID {str_id}")
                continue
            seen_ids.add(str_id)

            passage = record.get("passage", "")
            if not isinstance(passage, str) or not passage.strip():
                warnings.append(
                    f"SKIP {member_name}: blank/missing passage (ID={str_id})"
                )
                continue

            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                name = str_id  # deterministic fallback per TASK_CONTRACT

            link = record.get("link")
            if not isinstance(link, str) or not link.strip():
                link = None

            documents.append(
                LegalDocument(
                    id=str_id,
                    name=name,
                    passage=passage,
                    source_path=str(archive_path),
                    content_hash=content_hash,
                    link=link,
                    source_member=member_name,
                )
            )

    if warnings:
        import sys
        for w in warnings:
            print(f"[data_loader] WARNING: {w}", file=sys.stderr)

    if not documents:
        raise DataLoadError(
            f"No valid documents loaded from {archive_path}"
        )

    return documents
