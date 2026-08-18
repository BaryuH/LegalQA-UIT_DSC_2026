"""Acceptance tests for TASK 09 synthetic query contracts and filters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legal_rag.schemas import LegalDocument
from legal_rag.sedar_retrieval.corpus.hierarchy import nodes_to_passages
from legal_rag.sedar_retrieval.corpus.parse_legal import parse_legal_document
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    GeneratedQuery,
    SyntheticGenerationConfig,
    SyntheticGenerationErrorRecord,
    SyntheticQueryRecord,
    TemplateQueryGenerator,
    assign_document_splits,
    build_generator_prompt,
    build_synthetic_records,
    load_synthetic_records,
    validate_document_isolation,
    write_synthetic_records,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _passages(document_id: str) -> tuple:
    document = LegalDocument(
        id=document_id,
        name="Bộ luật Lao động 2019",
        passage=(
            "Điều 76. Nghỉ hằng năm\n"
            "1. Người lao động được nghỉ hằng năm theo quy định của pháp luật.\n"
        ),
        source_path="selected-contexts.zip",
        source_member=f"{document_id}.json",
        content_hash=f"hash-{document_id}",
    )
    nodes = parse_legal_document(document)
    return nodes_to_passages(nodes, levels=("article", "clause"))


def test_cli_template_backend_writes_pilot_artifacts(tmp_path: Path) -> None:
    document_id = next(
        str(index)
        for index in range(100)
        if assign_document_splits((str(index),), seed=42)[str(index)] == "train"
    )
    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        json.dumps(
            _passages(document_id)[0].model_dump(mode="json"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "task09"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT))
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts"
                / "sedar_retrieval"
                / "generate_synthetic_queries.py"
            ),
            "--passages",
            str(passages_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "template",
            "--query-types",
            "direct",
            "--target-accepted",
            "1",
            "--max-attempts",
            "1",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "NEEDS_MANUAL_AUDIT"
    assert manifest["gold_answers_used"] is False
    assert (
        len(
            (output_dir / "synthetic_queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 1
    )
    assert (output_dir / "rejections.jsonl").read_text(encoding="utf-8") == ""
    assert (output_dir / "generation_errors.jsonl").read_text(encoding="utf-8") == ""


def test_document_split_is_deterministic_and_disjoint() -> None:
    first = assign_document_splits(("d1", "d2", "d3"), seed=42)
    second = assign_document_splits(("d3", "d1", "d2"), seed=42)
    assert first == second
    assert set(first.values()).issubset({"train", "validation", "test"})


def test_template_generator_builds_schema_safe_record() -> None:
    passage = _passages("d1")[0]
    records, report = build_synthetic_records(
        (passage,),
        generator=TemplateQueryGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("direct",),
        ),
        document_splits={"d1": "train"},
    )
    assert report.accepted == 1
    assert len(records) == 1
    payload = records[0].model_dump()
    assert payload["source_document_id"] == "d1"
    assert "answer" not in payload
    assert passage.reader_text not in records[0].query


def test_prompt_includes_type_rules_and_source_metadata() -> None:
    prompt = build_generator_prompt(_passages("d1")[0], "citation_free")
    assert "one line only" in prompt
    assert "Điều, Khoản, Điểm" in prompt
    assert "Article: 76" in prompt
    assert "Bộ luật Lao động 2019" in prompt


@pytest.mark.parametrize(
    ("candidate", "reason"),
    (
        (
            "2. Người lao động được nghỉ hằng năm như thế nào?",
            "list_fragment",
        ),
        (
            "Người lao động được nghỉ hằng năm theo quy định của pháp luật.",
            "not_question",
        ),
        (
            "Theo Điều 76, người lao động được nghỉ hằng năm như thế nào?",
            "forbidden_citation",
        ),
    ),
)
def test_question_shape_and_type_filters_reject_bad_candidates(
    candidate: str,
    reason: str,
) -> None:
    passage = _passages("d1")[0]

    class CandidateGenerator:
        model = "test-generator"
        revision = "test-v2"

        def generate(self, passage, query_type):
            return GeneratedQuery(query=candidate, raw_generation=candidate)

    records, report = build_synthetic_records(
        (passage,),
        generator=CandidateGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("citation_free",),
        ),
        document_splits={"d1": "train"},
    )
    assert records == ()
    assert report.rejection_counts[reason] == 1


def test_known_generation_label_is_removed_before_acceptance() -> None:
    passage = _passages("d1")[0]

    class LabelledGenerator:
        model = "test-generator"
        revision = "test-v2"

        def generate(self, passage, query_type):
            query = (
                "Vietnamese legal question: Người lao động được nghỉ "
                "hằng năm như thế nào?"
            )
            return GeneratedQuery(query=query, raw_generation=query)

    records, report = build_synthetic_records(
        (passage,),
        generator=LabelledGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("direct",),
        ),
        document_splits={"d1": "train"},
    )
    assert report.accepted == 1
    assert records[0].query.startswith("Người lao động")
    assert "Vietnamese legal question" not in records[0].query


def test_wrong_citation_is_rejected() -> None:
    passage = _passages("d1")[0]

    class WrongCitationGenerator:
        model = "test-generator"
        revision = "test-v1"

        def generate(self, passage, query_type):
            return type(
                "Generated",
                (),
                {
                    "query": "Điều 99 quy định điều kiện nghỉ hằng năm như thế nào?",
                    "raw_generation": (
                        "Điều 99 quy định điều kiện nghỉ hằng năm như thế nào?"
                    ),
                },
            )()

    records, report = build_synthetic_records(
        (passage,),
        generator=WrongCitationGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("direct",),
        ),
        document_splits={"d1": "train"},
    )
    assert records == ()
    assert report.rejection_counts["wrong_citation"] == 1


def test_generation_errors_are_explicit_and_reasoning_is_not_persisted(
    tmp_path: Path,
) -> None:
    passage = _passages("d1")[0]

    class UnsafeGenerator:
        model = "test-generator"
        revision = "test-v1"

        def generate(self, passage, query_type):
            return GeneratedQuery(
                query=(
                    "<think>secret chain of thought</think>\n"
                    "Câu hỏi: Người lao động được nghỉ hằng năm như thế nào?"
                ),
                raw_generation=(
                    "<think>secret chain of thought</think>\n"
                    "Câu hỏi: Người lao động được nghỉ hằng năm như thế nào?"
                ),
            )

    records, report = build_synthetic_records(
        (passage,),
        generator=UnsafeGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("direct",),
        ),
        document_splits={"d1": "train"},
    )
    assert report.accepted == 1
    assert "secret" not in records[0].raw_generation
    assert "Câu hỏi:" not in records[0].raw_generation

    class FailingGenerator:
        model = "test-generator"
        revision = "test-v1"

        def generate(self, passage, query_type):
            raise RuntimeError("provider unavailable")

    errors: list[SyntheticGenerationErrorRecord] = []
    failed_records, failed_report = build_synthetic_records(
        (passage,),
        generator=FailingGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=1,
            max_attempts=1,
            query_types=("direct",),
        ),
        document_splits={"d1": "train"},
        generation_errors=errors,
    )
    assert failed_records == ()
    assert failed_report.generation_error_count == 1
    assert errors[0].error_message == "provider unavailable"

    records_path = tmp_path / "records.jsonl"
    write_synthetic_records(records_path, records)
    assert load_synthetic_records(records_path) == records


def test_near_duplicate_queries_are_rejected() -> None:
    passages = _passages("d1") + _passages("d2")

    class NearDuplicateGenerator:
        model = "test-generator"
        revision = "test-v1"
        calls = 0

        def generate(self, passage, query_type):
            self.calls += 1
            year = 2020 + self.calls
            query = f"Trong năm {year}, người lao động được nghỉ hằng năm như thế nào?"
            return type("Generated", (), {"query": query, "raw_generation": query})()

    records, report = build_synthetic_records(
        passages,
        generator=NearDuplicateGenerator(),
        config=SyntheticGenerationConfig(
            target_accepted=2,
            max_attempts=2,
            query_types=("citation_free",),
        ),
        document_splits={"d1": "train", "d2": "train"},
    )
    assert len(records) == 1
    assert report.rejection_counts["near_duplicate"] == 1


def test_document_isolation_detects_cross_split_conflict() -> None:
    common = {
        "synthetic_id": "s1",
        "query": "Người lao động được nghỉ hằng năm như thế nào?",
        "positive_passage_id": "p1",
        "source_document_id": "d1",
        "query_type": "direct",
        "legal_domain": "vietnamese_law",
        "generator_model": "test",
        "generator_revision": "v1",
        "prompt_version": "v1",
        "source_hash": "h",
        "quality_flags": ("heuristic_supportable",),
        "raw_generation": "Người lao động được nghỉ hằng năm như thế nào?",
    }
    records = (
        SyntheticQueryRecord(**common, source_split="train"),
        SyntheticQueryRecord(
            **{**common, "synthetic_id": "s2"},
            source_split="validation",
        ),
    )
    isolated, conflicts = validate_document_isolation(records)
    assert not isolated
    assert conflicts == {"d1": ("train", "validation")}
