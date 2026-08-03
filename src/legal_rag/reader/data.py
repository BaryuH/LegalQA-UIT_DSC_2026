"""Read-only ALQAC CSV and immutable split-manifest loading."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .types import ReaderCase

ReaderSplit = Literal["train", "validation", "test"]


class ReaderDataError(ValueError):
    """Raised when reader data cannot satisfy the declared contract."""


@dataclass(frozen=True, slots=True)
class ReaderDataset:
    cases: tuple[ReaderCase, ...]
    splits: dict[ReaderSplit, tuple[str, ...]]
    dataset_sha256: str
    split_manifest_sha256: str

    def cases_for_split(self, split: ReaderSplit) -> tuple[ReaderCase, ...]:
        by_id = {case.id: case for case in self.cases}
        return tuple(by_id[case_id] for case_id in self.splits[split])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reader_dataset(csv_path: Path, split_manifest_path: Path) -> ReaderDataset:
    """Load data without mutating it and verify IDs/hash/split disjointness."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Reader dataset not found: {csv_path}")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(
            f"Reader split manifest not found: {split_manifest_path}"
        )
    dataset_hash = _sha256(csv_path)
    cases: list[ReaderCase] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"context", "question", "answer"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ReaderDataError(
                "Reader CSV must contain context, question, and answer columns"
            )
        for row_index, row in enumerate(reader):
            cases.append(
                ReaderCase(
                    id=f"vilqa-{row_index}",
                    context=row["context"],
                    question=row["question"],
                    answer=row["answer"],
                )
            )
    if not cases:
        raise ReaderDataError("Reader CSV must contain at least one row")

    try:
        raw: object = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReaderDataError("Reader split manifest is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ReaderDataError("Reader split manifest must be a JSON object")
    if raw.get("dataset_sha256") != dataset_hash:
        raise ReaderDataError("Reader dataset hash does not match split manifest")
    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, dict):
        raise ReaderDataError("Reader split manifest must contain a splits object")

    known_ids = {case.id for case in cases}
    seen: set[str] = set()
    splits: dict[ReaderSplit, tuple[str, ...]] = {}
    for split in ("train", "validation", "test"):
        identifiers = raw_splits.get(split)
        if not isinstance(identifiers, list) or not identifiers:
            raise ReaderDataError(f"Reader split {split!r} must be a non-empty list")
        normalized = tuple(str(identifier) for identifier in identifiers)
        if len(set(normalized)) != len(normalized):
            raise ReaderDataError(f"Reader split {split!r} contains duplicate IDs")
        unknown = set(normalized) - known_ids
        if unknown:
            raise ReaderDataError(
                f"Reader split {split!r} contains unknown IDs: {sorted(unknown)}"
            )
        overlap = seen.intersection(normalized)
        if overlap:
            raise ReaderDataError(f"Reader splits overlap on IDs: {sorted(overlap)}")
        seen.update(normalized)
        splits[split] = normalized
    if seen != known_ids:
        raise ReaderDataError(
            f"Reader split coverage mismatch; missing={sorted(known_ids - seen)}"
        )
    return ReaderDataset(
        cases=tuple(cases),
        splits=splits,
        dataset_sha256=dataset_hash,
        split_manifest_sha256=_sha256(split_manifest_path),
    )


__all__ = ["ReaderDataError", "ReaderDataset", "ReaderSplit", "load_reader_dataset"]
