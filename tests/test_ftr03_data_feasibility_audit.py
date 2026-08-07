"""FTR-03 data feasibility and leakage audit acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.finetuned_reader.data_feasibility_audit import (
    audit_index_metadata_for_gold,
    audit_reference_access_policy,
    audit_split_file,
    audit_training_path_policy,
    cross_split_overlaps,
    normalize_question_text,
    run_data_feasibility_audit,
    run_retrieval_support_audit,
    training_overlap_remediation,
    write_audit_artifacts,
)
from legal_rag.generation import PromptBuilder
from legal_rag.retrieval import BM25Config, build_bm25_index
from legal_rag.schemas import LegalChunk, LegalDocument, LegalQuestion

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD = "Câu trả lời vàng không được vào truy vấn"


def _write_questions(path: Path, records: dict[str, dict[str, str]]) -> Path:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _chunk(chunk_id: str, document_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="ftr03-fixture.zip",
        source_member=f"context_{document_id}.json",
        section_label="Điều 1",
        raw_text=text,
        retrieval_text=text,
        content_hash=f"hash-{chunk_id}",
        chunker_version="legal-chunker-v1",
    )


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    _write_questions(
        data / "train.json",
        {
            "t1": {
                "question": "Nghỉ phép năm được quy định thế nào?",
                "answer": "Người lao động được nghỉ phép năm theo luật.",
            },
            "t2": {
                "question": "Tiền lương trả khi nào?",
                "answer": "Tiền lương được trả theo thỏa thuận hợp đồng.",
            },
        },
    )
    _write_questions(
        data / "warmup.json",
        {
            "w1": {
                "question": "Warmup câu hỏi pháp lý?",
                "answer": "Warmup câu trả lời pháp lý.",
            }
        },
    )
    return data


def test_valid_train_fixture(split_dir: Path) -> None:
    audit = audit_split_file(
        split_dir / "train.json", split="train", include_answers=True
    )
    assert audit.status == "present"
    assert audit.record_count == 2
    assert audit.duplicate_id_count == 0
    assert audit.blank_answer_count == 0
    assert audit.missing_answer_count == 0


def test_duplicate_id_detected(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(
        "{\n"
        '  "dup": {"question": "Câu một?", "answer": "A"},\n'
        '  "dup": {"question": "Câu hai?", "answer": "B"}\n'
        "}\n",
        encoding="utf-8",
    )
    audit = audit_split_file(path, split="train", include_answers=True)
    assert audit.duplicate_id_count >= 1
    assert audit.status == "error"


def test_blank_answer_counted(tmp_path: Path) -> None:
    path = tmp_path / "blank.json"
    path.write_text(
        json.dumps(
            {"b1": {"question": "Câu hỏi không trống?", "answer": "   "}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audit = audit_split_file(path, split="train", include_answers=True)
    assert audit.blank_answer_count >= 1
    assert audit.status == "error"


def test_cross_split_overlap(split_dir: Path) -> None:
    train = audit_split_file(
        split_dir / "train.json", split="train", include_answers=True
    )
    warmup = audit_split_file(
        split_dir / "warmup.json", split="warmup", include_answers=True
    )
    # Inject shared normalized question via a third file.
    shared = _write_questions(
        split_dir / "public-official.json",
        {
            "p1": {
                "question": "Nghỉ phép năm được quy định thế nào?",
                "answer": "should-not-load",
            }
        },
    )
    public = audit_split_file(shared, split="public", include_answers=False)
    overlaps = cross_split_overlaps(
        {"train": train, "warmup": warmup, "public": public}
    )
    assert overlaps["normalized_question_overlap_counts"]["public__train"] == 1
    assert overlaps["forbidden_overlap_total"] >= 1


def test_overlap_remediation_excludes_every_matching_train_duplicate(
    split_dir: Path,
) -> None:
    train = audit_split_file(
        split_dir / "train.json", split="train", include_answers=True
    )
    _write_questions(
        split_dir / "warmup.json",
        {
            "w1": {
                "question": train.questions_by_id["t1"],
                "answer": "Warmup cÃ¢u tráº£ lá»i phÃ¡p lÃ½.",
            }
        },
    )
    splits = {
        "train": train,
        "warmup": audit_split_file(
            split_dir / "warmup.json", split="warmup", include_answers=True
        ),
    }

    report, exclusions = training_overlap_remediation(
        splits,
        overlap_policy="exclude_and_record",
        remediation_id="fixture-remediation-v1",
    )

    assert report["status"] == "remediated"
    assert report["effective_train_count"] == 1
    assert report["effective_train_to_nontrain_overlap_case_count"] == 0
    assert [item.case_id for item in exclusions] == ["t1"]
    assert exclusions[0].reasons == ("normalized_question_overlap:warmup",)


def test_utf8_vietnamese_normalization() -> None:
    left = normalize_question_text("  Nghỉ phép năm  ")
    right = normalize_question_text("nghỉ phép năm")
    assert left == right


def test_gold_never_enters_retrieval_query(tmp_path: Path) -> None:
    chunks = (
        _chunk("c1", "d1", "Điều 1. Nghỉ phép năm theo Bộ luật Lao động."),
        _chunk("c2", "d2", "Điều 2. Tiền lương theo hợp đồng."),
    )
    documents = {
        chunk.document_id: LegalDocument(
            id=chunk.document_id,
            name=f"VB {chunk.document_id}",
            passage=chunk.raw_text,
            source_path=chunk.source_path,
            source_member=chunk.source_member,
            content_hash=f"src-{chunk.document_id}",
        )
        for chunk in chunks
    }
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "ftr03-cache",
        BM25Config(k1=1.5, b=0.75),
    )
    assert built.index is not None
    config = load_config(REPO_ROOT / "configs" / "frozen" / "hybrid_rag_b2.yaml")
    builder = PromptBuilder.from_config(
        config.prompts, prompt_dir=REPO_ROOT / "configs" / "prompts"
    )
    question = LegalQuestion(
        id="q1",
        question="Nghỉ phép năm được quy định thế nào?",
        answer=GOLD,
        split="train",
    )
    summary, cases = run_retrieval_support_audit(
        (question,),
        index=built.index,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        documents=documents,
        config=config,
        prompt_builder=builder,
    )
    assert summary["status"] == "ok"
    assert summary["gold_in_query_count"] == 0
    assert cases[0].query == question.question
    assert GOLD not in cases[0].query


def test_gold_never_enters_index_metadata(tmp_path: Path) -> None:
    chunks = (_chunk("c1", "d1", "Điều luật không chứa đáp án vàng."),)
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "ftr03-index",
        BM25Config(),
    )
    assert built.index is not None
    report = audit_index_metadata_for_gold(built.index, gold_answers=(GOLD,))
    assert report["pass"] is True
    assert report["forbidden_field_count"] == 0


def test_gold_in_index_text_fails(tmp_path: Path) -> None:
    chunks = (_chunk("c1", "d1", f"Đoạn chứa {GOLD} trong corpus."),)
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "ftr03-leaky",
        BM25Config(),
    )
    assert built.index is not None
    report = audit_index_metadata_for_gold(built.index, gold_answers=(GOLD,))
    assert report["pass"] is False
    assert report["gold_text_in_index_document_count"] == 1


def test_private_answer_access_blocked() -> None:
    report = audit_reference_access_policy()
    assert report["public_answer_load_blocked"] is True
    assert report["private_answer_load_blocked"] is True
    assert report["pass"] is True


def test_training_paths_train_only() -> None:
    assert audit_training_path_policy(("train",))["pass"] is True
    assert audit_training_path_policy(("train", "public"))["pass"] is False
    assert audit_training_path_policy(("private",))["pass"] is False


def test_deterministic_audit_ordering(tmp_path: Path, split_dir: Path) -> None:
    # Point a temporary repo-like tree at frozen config/prompts/manifest via symlink
    # is hard on Windows; call retrieval audit twice and compare serialized cases.
    chunks = (
        _chunk("c-b", "d1", "Điều B. Nghỉ phép năm theo luật."),
        _chunk("c-a", "d2", "Điều A. Tiền lương theo hợp đồng."),
    )
    built = build_bm25_index(
        list(chunks),
        tmp_path / "cache",
        "ftr03-order",
        BM25Config(),
    )
    assert built.index is not None
    config = load_config(REPO_ROOT / "configs" / "frozen" / "hybrid_rag_b2.yaml")
    builder = PromptBuilder.from_config(
        config.prompts, prompt_dir=REPO_ROOT / "configs" / "prompts"
    )
    questions = (
        LegalQuestion(
            id="z",
            question="Tiền lương?",
            answer="Theo hợp đồng.",
            split="train",
        ),
        LegalQuestion(
            id="a",
            question="Nghỉ phép?",
            answer="Theo luật.",
            split="train",
        ),
    )
    first, cases_a = run_retrieval_support_audit(
        questions,
        index=built.index,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        documents=None,
        config=config,
        prompt_builder=builder,
    )
    second, cases_b = run_retrieval_support_audit(
        reversed(questions),
        index=built.index,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        documents=None,
        config=config,
        prompt_builder=builder,
    )
    assert [case.id for case in cases_a] == ["a", "z"]
    assert [case.id for case in cases_b] == ["a", "z"]
    assert first == second


def test_write_artifacts_deterministic(tmp_path: Path) -> None:
    result = run_data_feasibility_audit(REPO_ROOT)
    paths_a = write_audit_artifacts(result, tmp_path / "a")
    paths_b = write_audit_artifacts(result, tmp_path / "b")
    assert paths_a["summary"].read_text(encoding="utf-8") == paths_b[
        "summary"
    ].read_text(encoding="utf-8")
    assert paths_a["leakage_report"].read_text(encoding="utf-8") == paths_b[
        "leakage_report"
    ].read_text(encoding="utf-8")
    assert paths_a["training_exclusions"].read_text(encoding="utf-8") == paths_b[
        "training_exclusions"
    ].read_text(encoding="utf-8")


def test_repository_audit_hard_stops_on_observed_cross_split_overlap() -> None:
    result = run_data_feasibility_audit(REPO_ROOT)
    assert result.hard_stop is True
    assert result.splits["warmup"]["status"] == "present"
    assert result.splits["train"]["status"] == "present"
    assert result.retrieval["status"] in {"blocked", "partial"}
    assert any("cross-split overlaps" in reason for reason in result.hard_stop_reasons)


def test_repository_audit_passes_with_approved_effective_train_remediation() -> None:
    result = run_data_feasibility_audit(
        REPO_ROOT,
        overlap_policy="exclude_and_record",
        overlap_remediation_id="ftr03-train-overlap-exclusion-v1",
    )

    assert result.status == "pass"
    assert result.hard_stop is False
    assert result.cross_split["forbidden_overlap_total"] == 855
    assert result.training_remediation["source_train_overlap_case_count"] == 391
    assert result.training_remediation["effective_train_count"] == 6609
    assert (
        result.training_remediation["effective_train_to_nontrain_overlap_case_count"]
        == 0
    )
    assert result.training_remediation["unremediated_nontraining_overlap_total"] == 80
