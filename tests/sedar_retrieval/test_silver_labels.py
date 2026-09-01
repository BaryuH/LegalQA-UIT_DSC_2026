"""Acceptance tests for document-scoped silver labels."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.eval.silver_label_audit import audit_silver_labels
from legal_rag.sedar_retrieval.eval.silver_labels import (
    SILVER_LABEL_SCHEMA_VERSION,
    build_silver_labels_from_answers,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _passage(
    passage_id: str,
    document_id: str,
    document_name: str,
    article_number: str,
    *,
    retrieval_level: Literal["article", "clause"] = "article",
) -> CanonicalPassage:
    text = f"Nội dung Điều {article_number}."
    if passage_id == "doc-a-art-12":
        text += "\u2028Bảo toàn line separator trong legal text."
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=document_id,
        article_id=f"{document_id}::art::{article_number}",
        retrieval_level=retrieval_level,
        document_name=document_name,
        article_number=article_number,
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=SourceProvenance(
            source_path=f"data/{document_id}.txt",
            document_id=document_id,
            content_hash=f"hash-{document_id}",
        ),
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "q-ambiguous": {"answer": "Theo Điều 12, nội dung được áp dụng."},
                "q-multi": {
                    "answer": (
                        "Theo Điều 12 Thông tư 55/2021/TT-BCA. "
                        "Điều 3 Nghị định 83/2017/NĐ-CP cũng được áp dụng."
                    )
                },
                "q-name": {
                    "answer": "Theo Điều 7 Bộ luật Lao động 2019, người lao động..."
                },
                "q-scoped": {
                    "answer": (
                        "Theo Điều 12 Thông tư 55/2021/TT-BCA, "
                        "quy định này được áp dụng."
                    )
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    passages_path = tmp_path / "passages.jsonl"
    passages = (
        _passage(
            "doc-a-art-12",
            "doc-a",
            "Thong-tu-55-2021-TT-BCA-huong-dan-466836",
            "12",
        ),
        _passage(
            "doc-a-clause-12",
            "doc-a",
            "Thong-tu-55-2021-TT-BCA-huong-dan-466836",
            "12",
            retrieval_level="clause",
        ),
        _passage(
            "doc-b-art-12",
            "doc-b",
            "Nghi-dinh-99-2022-ND-CP-khac-999999",
            "12",
        ),
        _passage(
            "doc-c-art-3",
            "doc-c",
            "Nghi-dinh-83-2017-ND-CP-huong-dan-838383",
            "3",
        ),
        _passage(
            "doc-d-art-7",
            "doc-d",
            "Bo-luat-lao-dong-2019-333670",
            "7",
        ),
    )
    passages_path.write_text(
        "".join(
            json.dumps(passage.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for passage in passages
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "labels-v2.jsonl"
    return questions_path, passages_path, output_path


def test_builder_scopes_articles_to_cited_document(tmp_path: Path) -> None:
    questions_path, passages_path, output_path = _write_fixture(tmp_path)

    stats = build_silver_labels_from_answers(
        questions_path=questions_path,
        passages_path=passages_path,
        output_path=output_path,
    )

    rows = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert stats == {
        "labeled": 3,
        "unlabeled": 1,
        "total": 4,
        "resolved_citations": 4,
        "ambiguous_citations": 1,
        "resolution_reason_counts": {
            "document_identity_not_found": 1,
            "resolved_document_name": 1,
            "resolved_document_number": 3,
        },
        "schema_version": SILVER_LABEL_SCHEMA_VERSION,
    }
    assert rows["q-scoped"]["relevant_ids"] == ["doc-a-art-12"]
    assert rows["q-scoped"]["provenance"] == "silver"
    assert rows["q-ambiguous"]["relevant_ids"] == []
    assert rows["q-ambiguous"]["provenance"] == "unlabeled"
    assert rows["q-ambiguous"]["resolution_reasons"] == {
        "document_identity_not_found": 1
    }
    assert rows["q-name"]["relevant_ids"] == ["doc-d-art-7"]
    assert rows["q-multi"]["relevant_ids"] == ["doc-a-art-12", "doc-c-art-3"]
    assert all(
        row["schema_version"] == SILVER_LABEL_SCHEMA_VERSION for row in rows.values()
    )
    audit = audit_silver_labels(
        labels_path=output_path,
        passages_path=passages_path,
    ).as_dict()
    assert audit["resolution_reason_counts"] == {
        "document_identity_not_found": 1,
        "resolved_document_name": 1,
        "resolved_document_number": 3,
    }
    assert audit["unresolved_query_sample"][0]["query_id"] == "q-ambiguous"
    assert all("answer" not in sample for sample in audit["unresolved_query_sample"])
    assert "Theo Điều" not in json.dumps(audit, ensure_ascii=False)


def test_builder_keeps_articleless_answer_unlabeled(tmp_path: Path) -> None:
    _, passages_path, _ = _write_fixture(tmp_path)
    questions_path = tmp_path / "articleless-questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "q-no-article": {
                    "answer": (
                        "Theo Thông tư 55/2021/TT-BCA, quy định này được áp dụng."
                    )
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "articleless-labels-v2.jsonl"

    stats = build_silver_labels_from_answers(
        questions_path=questions_path,
        passages_path=passages_path,
        output_path=output_path,
    )
    row = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert stats["labeled"] == 0
    assert stats["unlabeled"] == 1
    assert stats["resolved_citations"] == 0
    assert stats["ambiguous_citations"] == 0
    assert row["provenance"] == "unlabeled"
    assert row["note"] == "no_article_citation"
    assert row["relevant_ids"] == []
    assert row["resolution_reasons"] == {"no_article_citation": 1}


def test_builder_rejects_empty_passage_corpus(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text('{"q": {"answer": "Theo Điều 1."}}\n', encoding="utf-8")
    passages_path = tmp_path / "empty.jsonl"
    passages_path.write_text("", encoding="utf-8")

    try:
        build_silver_labels_from_answers(
            questions_path=questions_path,
            passages_path=passages_path,
            output_path=tmp_path / "labels.jsonl",
        )
    except ValueError as exc:
        assert "Passage corpus is empty" in str(exc)
    else:
        raise AssertionError("empty passage corpus must fail closed")


def test_audit_exposes_global_article_mapping_pattern(tmp_path: Path) -> None:
    _, passages_path, _ = _write_fixture(tmp_path)
    labels_path = tmp_path / "legacy-labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "query_id": "q-legacy",
                "relevant_ids": ["doc-a-art-12", "doc-b-art-12"],
                "provenance": "silver",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_silver_labels(
        labels_path=labels_path,
        passages_path=passages_path,
    ).as_dict()

    assert report["multi_document_query_count"] == 1
    assert report["document_count_histogram"] == {2: 1}
    assert report["selected_from_global_article_first_three_rate"] == 1.0
    assert report["gold_answer_text_written"] is False


def test_audit_cli_writes_evaluation_only_report(tmp_path: Path) -> None:
    _, passages_path, _ = _write_fixture(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "relevant_ids": ["doc-a-art-12"],
                "provenance": "silver",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts"
                / "sedar_retrieval"
                / "audit_silver_labels.py"
            ),
            "--labels",
            str(labels_path),
            "--passages",
            str(passages_path),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sedar-silver-label-audit-v1"
    assert payload["gold_answer_text_written"] is False
    assert "Theo Điều" not in output_path.read_text(encoding="utf-8")
