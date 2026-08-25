"""Acceptance tests for TASK 20 SEDAR e2e runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from legal_rag.finetuned_reader.checkpoint import ValidatedCheckpoint
from legal_rag.finetuned_reader.contracts import GenerationResult
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.schemas import PackedEvidence
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage, SourceProvenance
from legal_rag.sedar_retrieval.e2e.runner import SedarE2EConfig, run_sedar_e2e
from legal_rag.sedar_retrieval.evidence.passage_packer import (
    PassageEvidenceConfig,
    load_retrieval_rankings,
    pack_passage_retrieval_evidence,
)


def _passage(passage_id: str, article: str, clause: str, text: str) -> CanonicalPassage:
    document_id = "law-1"
    source = SourceProvenance(
        source_path="data/law-1.txt",
        document_id=document_id,
        content_hash=f"hash-{passage_id}",
    )
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=document_id,
        article_id=f"{document_id}:article:{article}",
        clause_id=f"{document_id}:article:{article}:clause:{clause}",
        retrieval_level="clause",
        document_name="Luật Lao động",
        article_number=article,
        clause_number=clause,
        status="effective",
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=source,
    )


def test_load_retrieval_rankings_and_pack_evidence(tmp_path: Path) -> None:
    retrieval = tmp_path / "dense.jsonl"
    retrieval.write_text(
        json.dumps(
            {
                "query_id": "101515",
                "ranked_ids": ["p-1", "p-2"],
                "scores": [
                    {"passage_id": "p-1", "dense": 0.9, "rank": 1, "source": "dense"},
                    {"passage_id": "p-2", "dense": 0.4, "rank": 2, "source": "dense"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rankings = load_retrieval_rankings(retrieval)
    passages = {
        "p-1": _passage("p-1", "76", "1", "Người lao động được nghỉ hằng năm."),
        "p-2": _passage("p-2", "77", "1", "Thời gian nghỉ hằng năm do thỏa thuận."),
    }
    packed = pack_passage_retrieval_evidence(
        rankings["101515"],
        passages,
        config=PassageEvidenceConfig(evidence_top_k=2),
    )
    assert isinstance(packed, PackedEvidence)
    assert packed.included_ids == ("p-1", "p-2")
    assert "Điều 76" in packed.rendered_text


@dataclass(frozen=True, slots=True)
class _FakeGenerator:
    checkpoint: ValidatedCheckpoint
    prompt_builder: GenerativePromptBuilder
    max_new_tokens: int = 32

    def generate(self, question: str, evidence: PackedEvidence) -> GenerationResult:
        prompt = self.prompt_builder.build_inference(question, evidence)
        raw = "Theo Điều 76, người lao động được nghỉ hằng năm."
        return GenerationResult(
            raw_answer=raw,
            cleaned_answer=raw,
            model="mock-model",
            model_version="mock-v1",
            latency_ms=1.0,
            prompt_hash=prompt.sha256,
            metadata={"mock": True},
        )


def _fake_generator(tmp_path: Path) -> _FakeGenerator:
    manifest = {
        "profile": "sedar_sft",
        "type": "generative_sft_reader",
        "base_model": "models/mock",
        "base_revision": "abc",
        "tokenizer": "models/mock",
        "adapter_type": "qlora",
        "target_modules": ["q_proj"],
        "adapter_hash": "adapter-hash",
        "dataset_manifest_hash": "ds",
        "retrieval_config_hash": "ret",
        "index_fingerprint": "idx",
        "prompt_hash": "prompt",
        "seed": 42,
        "best_checkpoint_criterion": "loss",
    }
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = ValidatedCheckpoint(
        checkpoint_dir=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_hash="manifest-hash",
        adapter_hash="adapter-hash",
    )
    prompt_builder = GenerativePromptBuilder.from_files(
        Path("configs/prompts/sedar_sft_train_v1.txt"),
        Path("configs/prompts/sedar_sft_infer_v1.txt"),
        version="sedar-sft-v1",
    )
    return _FakeGenerator(checkpoint=checkpoint, prompt_builder=prompt_builder)


def test_run_sedar_e2e_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    included_ids = ["101515", "101683"]
    manifest = {
        "policy_id": "sedar-warmup-public-exclusion-v1",
        "included_count": len(included_ids),
        "included_ids": included_ids,
        "excluded_count": 0,
        "excluded_ids": [],
        "included_ids_hash": "included-hash",
        "excluded_ids_hash": "excluded-hash",
        "source_warmup_sha256": "warmup-hash",
        "source_public_sha256": "public-hash",
        "source_warmup_count": 500,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    questions = {item: {"question": f"Câu hỏi {item}?"} for item in included_ids}
    questions_path = tmp_path / "warmup.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        "\n".join(
            json.dumps(
                _passage(f"p-{item}", "76", "1", "Nội dung pháp luật.").model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
            )
            for item in included_ids
        )
        + "\n",
        encoding="utf-8",
    )

    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": item,
                    "ranked_ids": [f"p-{item}"],
                    "scores": [
                        {
                            "passage_id": f"p-{item}",
                            "dense": 0.8,
                            "rank": 1,
                            "source": "dense",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            for item in included_ids
        )
        + "\n",
        encoding="utf-8",
    )

    config = SedarE2EConfig(
        retrieval_path=retrieval_path,
        passages_path=passages_path,
        questions_path=questions_path,
        manifest_path=manifest_path,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_manifest=None,
        inference_prompt_path=repo_root / "configs/prompts/sedar_sft_infer_v1.txt",
        train_prompt_path=repo_root / "configs/prompts/sedar_sft_train_v1.txt",
        prompt_version="sedar-sft-v1",
        output_dir=tmp_path / "outputs",
        retrieval_variant="dense_zero_shot",
        evidence=PassageEvidenceConfig(),
        max_new_tokens=32,
        stop_sequences=(),
        device="cpu",
        load_in_4bit=False,
        repo_root=repo_root,
    )
    (tmp_path / "checkpoint").mkdir()

    result = run_sedar_e2e(
        config,
        generator_factory=lambda: _fake_generator(tmp_path / "checkpoint"),
        run_id="task20_test_smoke",
    )
    assert result.prediction_count == 2
    predictions = (result.output_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert "101515" in predictions
    summary = json.loads((result.output_dir / "run_summary.json").read_text())
    assert summary["method"] == "sedar_sft"
    assert summary["retrieval_variant"] == "dense_zero_shot"


def test_run_sedar_e2e_cli_dry_import() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts/sedar_retrieval/run_sedar_e2e.py"
    assert script.is_file()
