"""JSONL readers that preserve legal text containing Unicode line separators."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def iter_jsonl_lines(path: str | Path) -> Iterator[str]:
    """Yield physical JSONL lines without splitting on U+2028/U+2029."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                yield line


def load_jsonl_records(path: str | Path) -> list[dict[str, object]]:
    """Load JSON objects from a JSONL file."""

    return [json.loads(line) for line in iter_jsonl_lines(path)]


def count_jsonl_records(path: str | Path) -> int:
    """Count non-empty JSONL records."""

    return sum(1 for _ in iter_jsonl_lines(path))
