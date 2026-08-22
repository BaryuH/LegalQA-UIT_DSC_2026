"""Acceptance tests for TASK 11 retriever LoRA scaffold."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.cuda_policy import CudaDeferredError
from legal_rag.sedar_retrieval.retrieval.dense import format_instruct_query
from legal_rag.sedar_retrieval.training.hard_negatives import (
    HardNegative,
    HardNegativeRecord,
    write_jsonl_models,
)
from legal_rag.sedar_retrieval.training.retriever_lora import (
    RetrieverLoRAConfig,
    RetrieverLoRATrainingError,
    build_retriever_dataset,
    build_retriever_examples,
    check_entry_gate,
    split_retriever_examples,
)
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    SyntheticQueryRecord,
    write_synthetic_records,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _passage(
    passage_id: str,
    document_id: str,
    article_number: str,
    clause_number: str,
    text: str,
) -> CanonicalPassage:
    source = SourceProvenance(
        source_path=f"data/{document_id}.txt",
        document_id=document_id,
        content_hash=f"hash-{passage_id}",
    )
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=document_id,
        article_id=f"{document_id}:article:{article_number}",
        clause_id=f"{document_id}:article:{article_number}:clause:{clause_number}",
        retrieval_level="clause",
        document_name="Bộ luật Lao động 2019",
        article_number=article_number,
        clause_number=clause_number,
        status="effective",
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=source,
    )


def _synthetic_record(
    synthetic_id: str,
    *,
    document_id: str,
    positive_passage_id: str,
    query: str,
) -> SyntheticQueryRecord:
    return SyntheticQueryRecord(
        synthetic_id=synthetic_id,
        query=query,
        positive_passage_id=positive_passage_id,
        source_document_id=document_id,
        query_type="direct",
        legal_domain="vietnamese_law",
        generator_model="test",
        generator_revision="test-v1",
        prompt_version="test-prompt-v1",
        source_hash=f"hash-{positive_passage_id}",
        quality_flags=("question_form_checked",),
        raw_generation=query,
        source_split="train",
    )


def _hard_negative_record(
    synthetic: SyntheticQueryRecord,
    *,
    negative_ids: tuple[str, ...],
) -> HardNegativeRecord:
    negatives = tuple(
        HardNegative(
            negative_passage_id=negative_id,
            negative_document_id=synthetic.source_document_id,
            negative_article_id=f"{synthetic.source_document_id}:article:1",
            negative_clause_id=f"{synthetic.source_document_id}:article:1:clause:1",
            negative_source_hash=f"hash-{negative_id}",
            negative_category="A",
            mined_from="bm25",
            rank=index + 1,
            retrieval_score=0.5 - index * 0.01,
            hardness_score=0.4,
            potential_false_negative=False,
        )
        for index, negative_id in enumerate(negative_ids)
    )
    return HardNegativeRecord(
        hard_negative_id=f"hn-{synthetic.synthetic_id}",
        synthetic_id=synthetic.synthetic_id,
        query=synthetic.query,
        query_type=synthetic.query_type,
        positive_passage_id=synthetic.positive_passage_id,
        source_document_id=synthetic.source_document_id,
        positive_source_hash=synthetic.source_hash,
        negatives=negatives,
        source_split="train",
    )


def test_build_retriever_examples_pairs_synthetic_and_hard_negatives() -> None:
    positive = _passage("p-pos", "doc-a", "76", "1", "Positive passage text.")
    negative_a = _passage("p-neg-a", "doc-b", "77", "1", "Hard negative A.")
    negative_b = _passage("p-neg-b", "doc-c", "78", "1", "Hard negative B.")
    synthetic = _synthetic_record(
        "syn-1",
        document_id="doc-a",
        positive_passage_id="p-pos",
        query="Người lao động được nghỉ hằng năm thế nào?",
    )
    hard_record = _hard_negative_record(
        synthetic,
        negative_ids=("p-neg-a", "p-neg-b"),
    )
    passages = {
        passage.passage_id: passage
        for passage in (positive, negative_a, negative_b)
    }

    examples = build_retriever_examples(
        synthetic_records=(synthetic,),
        hard_negative_records=(hard_record,),
        passages=passages,
        config=RetrieverLoRAConfig(),
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.synthetic_id == "syn-1"
    assert example.positive_passage == "Positive passage text."
    assert example.hard_negative_passages == ("Hard negative A.", "Hard negative B.")
    assert example.instruct_query == format_instruct_query(synthetic.query)


def test_split_retriever_examples_respects_document_isolation() -> None:
    examples = tuple(
        build_retriever_examples(
            synthetic_records=(
                _synthetic_record(
                    f"syn-{index}",
                    document_id=f"doc-{index}",
                    positive_passage_id=f"p-{index}",
                    query=f"Câu hỏi {index}?",
                )
                for index in range(20)
            ),
            hard_negative_records=tuple(
                _hard_negative_record(
                    _synthetic_record(
                        f"syn-{index}",
                        document_id=f"doc-{index}",
                        positive_passage_id=f"p-{index}",
                        query=f"Câu hỏi {index}?",
                    ),
                    negative_ids=(f"n-{index}-a", f"n-{index}-b"),
                )
                for index in range(20)
            ),
            passages={
                f"p-{index}": _passage(
                    f"p-{index}",
                    f"doc-{index}",
                    str(index),
                    "1",
                    f"Text {index}",
                )
                for index in range(20)
            }
            | {
                f"n-{index}-{suffix}": _passage(
                    f"n-{index}-{suffix}",
                    f"doc-{index}-neg",
                    str(index),
                    suffix,
                    f"Negative {index}-{suffix}",
                )
                for index in range(20)
                for suffix in ("a", "b")
            },
            config=RetrieverLoRAConfig(
                validation_fraction=0.1,
                test_fraction=0.1,
                seed=42,
            ),
        )
    )

    train, validation = split_retriever_examples(
        examples,
        seed=42,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    train_docs = {example.source_document_id for example in train}
    val_docs = {example.source_document_id for example in validation}
    assert train_docs.isdisjoint(val_docs)
    assert train
    assert validation


def test_check_entry_gate_requires_cuda_when_not_dry_run(tmp_path) -> None:
    synthetic = tmp_path / "synthetic.jsonl"
    hard_negatives = tmp_path / "hard_negatives.jsonl"
    passages = tmp_path / "passages.jsonl"
    synthetic.write_text("{}\n", encoding="utf-8")
    hard_negatives.write_text("{}\n", encoding="utf-8")
    passages.write_text("{}\n", encoding="utf-8")

    with patch(
        "legal_rag.sedar_retrieval.training.retriever_lora.require_dense_encode",
        side_effect=CudaDeferredError("no cuda"),
    ):
        with pytest.raises(CudaDeferredError):
            check_entry_gate(
                synthetic_path=synthetic,
                hard_negatives_path=hard_negatives,
                passages_path=passages,
                require_cuda=True,
            )


def test_build_retriever_dataset_fails_on_missing_hard_negatives(tmp_path) -> None:
    positive = _passage("p-1", "doc-1", "76", "1", "Positive text.")
    synthetic = _synthetic_record(
        "syn-1",
        document_id="doc-1",
        positive_passage_id="p-1",
        query="Câu hỏi thử nghiệm?",
    )
    synthetic_path = tmp_path / "synthetic.jsonl"
    write_synthetic_records(synthetic_path, (synthetic,))
    hard_path = tmp_path / "hard_negatives.jsonl"
    hard_path.write_text("", encoding="utf-8")

    with pytest.raises(RetrieverLoRATrainingError, match="Missing hard negatives"):
        build_retriever_dataset(
            synthetic_path=synthetic_path,
            hard_negatives_path=hard_path,
            passages=(positive,),
        )


def test_train_cli_dry_run_writes_manifest(tmp_path) -> None:
    positive = _passage("p-1", "doc-1", "76", "1", "Positive text.")
    negative_a = _passage("p-2", "doc-2", "77", "1", "Negative A.")
    negative_b = _passage("p-3", "doc-3", "78", "1", "Negative B.")
    synthetic = _synthetic_record(
        "syn-1",
        document_id="doc-1",
        positive_passage_id="p-1",
        query="Người lao động được nghỉ hằng năm thế nào?",
    )
    hard_record = _hard_negative_record(
        synthetic,
        negative_ids=("p-2", "p-3"),
    )
    synthetic_path = tmp_path / "synthetic.jsonl"
    hard_path = tmp_path / "hard_negatives.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    write_synthetic_records(synthetic_path, (synthetic,))
    write_jsonl_models(hard_path, (hard_record,), overwrite=True)
    passages_path.write_text(
        json.dumps(positive.model_dump(mode="json"), ensure_ascii=False)
        + "\n"
        + json.dumps(negative_a.model_dump(mode="json"), ensure_ascii=False)
        + "\n"
        + json.dumps(negative_b.model_dump(mode="json"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "task11"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT))
    )
    train_script = (
        REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "train_retriever_lora.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--synthetic",
            str(synthetic_path),
            "--hard-negatives",
            str(hard_path),
            "--passages",
            str(passages_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_path = output_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "DEFERRED_GPU"
    assert manifest["train_count"] == 1
