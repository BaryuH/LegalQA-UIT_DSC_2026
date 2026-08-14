"""Regression tests for JSONL readers with Unicode line separators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.sedar_retrieval.io.jsonl import (
    count_jsonl_records,
    iter_jsonl_lines,
    load_jsonl_records,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def test_iter_jsonl_lines_preserves_unicode_line_separator(tmp_path: Path) -> None:
    text = "Điều 1\u2028khoản a"
    payload = {"passage_id": "p1", "text": text}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = list(iter_jsonl_lines(path))
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == text


def test_load_passages_jsonl_with_unicode_line_separator(tmp_path: Path) -> None:
    raw_text = "Điều 76\u2028Khoản 1. Thỏa ước lao động."
    row = {
        "passage_id": "doc1::article::76",
        "document_id": "doc1",
        "retrieval_level": "article",
        "raw_text": raw_text,
        "reader_text": raw_text,
        "retrieval_text": raw_text,
        "article_number": "76",
        "clause_number": None,
        "point_label": None,
        "source": {
            "source_path": "zip",
            "source_member": "doc1.json",
            "document_id": "doc1",
            "content_hash": "abc123",
        },
    }
    path = tmp_path / "passages.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    passages = load_passages_jsonl(str(path))
    assert len(passages) == 1
    assert passages[0].raw_text == raw_text


def test_count_and_load_jsonl_records(tmp_path: Path) -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert count_jsonl_records(path) == 2
    assert load_jsonl_records(path) == rows


def test_splitlines_breaks_unicode_line_separator(tmp_path: Path) -> None:
    text = "Điều 1\u2028khoản a"
    payload = {"text": text}
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
