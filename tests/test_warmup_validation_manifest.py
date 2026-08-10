"""VAL-00 clean warmup validation manifest acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.finetuned_reader.split_remediation import normalize_question_text
from legal_rag.finetuned_reader.warmup_validation import (
    POLICY_ID,
    WarmupValidationError,
    build_clean_warmup_validation,
    derive_warmup_public_exclusions,
    write_clean_warmup_validation_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_questions(path: Path, records: dict[str, dict[str, str]]) -> Path:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    _write_questions(
        data / "warmup.json",
        {
            "w_keep": {
                "question": "Câu hỏi warmup không trùng?",
                "answer": "Warmup gold keep",
            },
            "w_id": {
                "question": "Câu hỏi trùng ID với public?",
                "answer": "Warmup gold id",
            },
            "w_norm": {
                "question": "  Nghỉ phép năm được quy định thế nào?  ",
                "answer": "Warmup gold norm",
            },
        },
    )
    _write_questions(
        data / "public-official.json",
        {
            "w_id": {
                "question": "Câu hỏi trùng ID với public?",
                "answer": "PUBLIC ANSWER MUST NOT AFFECT",
            },
            "p_norm": {
                "question": "nghỉ phép năm được quy định thế nào?",
                "answer": "Another public answer",
            },
            "p_other": {
                "question": "Câu hỏi public khác hẳn?",
                "answer": "Public other",
            },
        },
    )
    return data


def test_exact_id_overlap_excluded(fixture_dir: Path) -> None:
    result = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    excluded = {
        item["case_id"]: item["reason"] for item in result.overlap_report["excluded"]
    }
    assert "w_id" in excluded
    assert excluded["w_id"] in {
        "PUBLIC_ID_OVERLAP",
        "PUBLIC_ID_AND_QUESTION_OVERLAP",
    }
    assert "w_id" not in result.included_ids


def test_normalized_question_overlap_different_id_excluded(fixture_dir: Path) -> None:
    result = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    excluded = {
        item["case_id"]: item["reason"] for item in result.overlap_report["excluded"]
    }
    assert excluded["w_norm"] == "PUBLIC_NORMALIZED_QUESTION_OVERLAP"
    assert "w_norm" not in result.included_ids


def test_different_question_included(fixture_dir: Path) -> None:
    result = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    assert "w_keep" in result.included_ids
    assert result.manifest["included_count"] == 1
    assert result.overlap_report["union_excluded_count"] == 2


def test_duplicate_warmup_id_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(
        "{\n"
        '  "dup": {"question": "Câu một?", "answer": "A"},\n'
        '  "dup": {"question": "Câu hai?", "answer": "B"}\n'
        "}\n",
        encoding="utf-8",
    )
    public = _write_questions(
        tmp_path / "public-official.json",
        {"p1": {"question": "Khác?", "answer": "x"}},
    )
    with pytest.raises(WarmupValidationError, match="duplicate"):
        build_clean_warmup_validation(warmup_path=path, public_path=public)


def test_blank_warmup_question_fail_closed(tmp_path: Path) -> None:
    warmup = _write_questions(
        tmp_path / "warmup.json",
        {"w1": {"question": "   ", "answer": "gold"}},
    )
    public = _write_questions(
        tmp_path / "public-official.json",
        {"p1": {"question": "Khác?", "answer": "x"}},
    )
    with pytest.raises(WarmupValidationError, match="blank"):
        build_clean_warmup_validation(warmup_path=warmup, public_path=public)


def test_public_answer_does_not_affect_inclusion(fixture_dir: Path) -> None:
    first = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    public_path = fixture_dir / "public-official.json"
    payload = json.loads(public_path.read_text(encoding="utf-8"))
    for record in payload.values():
        record["answer"] = "COMPLETELY DIFFERENT PUBLIC GOLD"
    public_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=public_path,
    )
    assert first.included_ids == second.included_ids
    assert first.included_ids_hash == second.included_ids_hash
    assert first.excluded_ids_hash == second.excluded_ids_hash


def test_warmup_answer_does_not_affect_inclusion(fixture_dir: Path) -> None:
    first = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    warmup_path = fixture_dir / "warmup.json"
    payload = json.loads(warmup_path.read_text(encoding="utf-8"))
    for record in payload.values():
        record["answer"] = "COMPLETELY DIFFERENT WARMUP GOLD"
    warmup_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second = build_clean_warmup_validation(
        warmup_path=warmup_path,
        public_path=fixture_dir / "public-official.json",
    )
    assert first.included_ids == second.included_ids
    assert first.included_ids_hash == second.included_ids_hash


def test_changing_only_answer_identical_manifest(fixture_dir: Path) -> None:
    baseline = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    for name in ("warmup.json", "public-official.json"):
        path = fixture_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.values():
            record["answer"] = f"mutated-{name}"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    mutated = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    # Source file hashes change, but selection identity must not.
    assert baseline.included_ids == mutated.included_ids
    assert baseline.included_ids_hash == mutated.included_ids_hash
    assert baseline.excluded_ids_hash == mutated.excluded_ids_hash
    assert baseline.manifest["included_ids"] == mutated.manifest["included_ids"]
    assert baseline.manifest["policy_id"] == POLICY_ID


def test_deterministic_ordering_and_hash(fixture_dir: Path) -> None:
    first = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    second = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    assert list(first.included_ids) == sorted(first.included_ids)
    assert first.manifest == second.manifest
    assert first.overlap_report == second.overlap_report
    assert first.included_ids_hash == second.included_ids_hash


def test_unicode_normalization_matches_ftr03() -> None:
    left = normalize_question_text("  Nghỉ phép năm  ")
    right = normalize_question_text("nghỉ phép năm")
    assert left == right
    exclusions = derive_warmup_public_exclusions(
        {"w1": "  Nghỉ phép năm  "},
        {"p1": "nghỉ phép năm"},
    )
    assert len(exclusions) == 1
    assert exclusions[0].reason == "PUBLIC_NORMALIZED_QUESTION_OVERLAP"


def test_manifest_contains_no_reference_text(fixture_dir: Path) -> None:
    result = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    serialized = json.dumps(
        {
            "manifest": result.manifest,
            "overlap_report": result.overlap_report,
            "audit": result.audit,
        },
        ensure_ascii=False,
    )
    assert "Warmup gold" not in serialized
    assert "PUBLIC ANSWER" not in serialized
    assert "answer" not in result.manifest
    assert result.audit["answers_used_for_selection"] is False
    assert result.audit["public_answers_materialized"] is False


def test_write_artifacts(tmp_path: Path, fixture_dir: Path) -> None:
    result = build_clean_warmup_validation(
        warmup_path=fixture_dir / "warmup.json",
        public_path=fixture_dir / "public-official.json",
    )
    paths = write_clean_warmup_validation_artifacts(result, tmp_path / "validation")
    assert paths["manifest"].is_file()
    assert paths["overlap_report"].is_file()
    assert paths["audit"].is_file()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["included_ids_hash"] == result.included_ids_hash


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "warmup.json").is_file()
    or not (REPO_ROOT / "data" / "public-official.json").is_file(),
    reason="source warmup/public files unavailable",
)
def test_repo_source_build_is_deterministic_and_immutable() -> None:
    warmup = REPO_ROOT / "data" / "warmup.json"
    public = REPO_ROOT / "data" / "public-official.json"
    before_w = _sha256_file(warmup)
    before_p = _sha256_file(public)
    first = build_clean_warmup_validation(warmup_path=warmup, public_path=public)
    second = build_clean_warmup_validation(warmup_path=warmup, public_path=public)
    after_w = _sha256_file(warmup)
    after_p = _sha256_file(public)
    assert before_w == after_w == first.audit["source_warmup_sha256_after"]
    assert before_p == after_p == first.audit["source_public_sha256_after"]
    assert first.included_ids_hash == second.included_ids_hash
    assert first.manifest["source_warmup_count"] == 500
    assert first.overlap_report["exact_id_overlap_count"] == 40
    # Recomputed from FTR-03 normalizer; not hard-coded into the builder.
    assert (
        first.manifest["included_count"]
        == 500 - first.overlap_report["union_excluded_count"]
    )
    assert first.audit["source_hashes_unchanged"] is True
