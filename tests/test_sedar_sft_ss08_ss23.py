"""Offline tests for SEDAR-SFT SS-08..SS-23 scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.schemas import PackedEvidence, RetrievalHit
from legal_rag.sedar_sft.ablation import build_ablation_plan, write_ablation_plan
from legal_rag.sedar_sft.analyzer import analyze_requirements
from legal_rag.sedar_sft.checkpoint import (
    build_sedar_manifest_template,
    validate_sedar_checkpoint,
)
from legal_rag.sedar_sft.evidence_profile import build_evidence_profile
from legal_rag.sedar_sft.inference_baseline import run_sft_only_baseline
from legal_rag.sedar_sft.observability import GpuObservability, write_observability
from legal_rag.sedar_sft.preflight import run_canonical_train_preflight
from legal_rag.sedar_sft.promotion import build_promotion_freeze, write_promotion_freeze
from legal_rag.sedar_sft.review import build_adversarial_review, write_adversarial_review
from legal_rag.sedar_sft.runtime import run_sedar_runtime
from legal_rag.sedar_sft.verifier import verify_draft
from legal_rag.sedar_sft.draft import attach_draft_attribution


REPO = Path(__file__).resolve().parents[1]


def _packed(text: str = "Theo Điều 1 của 12/2020/NĐ-CP ngày 01/01/2020.") -> PackedEvidence:
    hit = RetrievalHit(
        chunk_id="c1",
        document_id="d1",
        source_path="t",
        rank=1,
        bm25_score=1.0,
    )
    return PackedEvidence(
        included_ids=("c1",),
        dropped_ids=(),
        truncated_ids=(),
        included_hits=(hit,),
        rendered_text=text,
        dropped_reasons={},
        metadata={},
    )


def test_ss08_preflight_blocks_without_gpu_auth() -> None:
    report = run_canonical_train_preflight(
        "configs/sedar_sft_train.yaml",
        repo_root=REPO,
        authorize_gpu_execution=False,
    )
    assert report.status == "blocked"
    assert "gpu_execution_deferred_by_operator" in report.blockers


def test_ss08_preflight_accepts_manifest_verify_file_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from legal_rag.sedar_sft import preflight as preflight_mod

    def fake_verify(repo_root: Path, manifest_path: Path) -> int:
        assert manifest_path == repo_root / "artifacts/data-baseline/manifest.json"
        return 4

    monkeypatch.setattr(
        "scripts.verify_data_manifest.verify_manifest",
        fake_verify,
    )
    preflight_mod._verify_source_manifest(REPO)


def test_ss09_sedar_checkpoint_validator(tmp_path: Path) -> None:
    ckpt = tmp_path / "run1"
    adapter = ckpt / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    from legal_rag.sedar_sft.checkpoint import hash_directory

    adapter_hash = hash_directory(adapter)
    base = {
        "profile": "sedar_sft",
        "type": "generative_sft_reader",
        "base_model": "models/sedar_sft/vilegalqwen3-1.7b-base",
        "base_revision": "abc123",
        "tokenizer": "models/sedar_sft/vilegalqwen3-1.7b-base",
        "adapter_type": "qlora",
        "target_modules": ["q_proj", "v_proj"],
        "adapter_hash": adapter_hash,
        "dataset_manifest_hash": "ds",
        "retrieval_config_hash": "ret",
        "index_fingerprint": "idx",
        "prompt_hash": "prompt",
        "seed": 42,
        "best_checkpoint_criterion": "loss",
    }
    manifest = build_sedar_manifest_template(base=base)
    (ckpt / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    validated = validate_sedar_checkpoint(ckpt)
    assert validated.adapter_hash == adapter_hash
    assert validated.sedar_fields["attention_backend"] == "sdpa"


def test_ss11_to_ss19_runtime_state_machine() -> None:
    evidence = _packed()
    result = run_sedar_runtime(
        question="Trách nhiệm hiện hành gồm những gì?",
        evidence=evidence,
        generate_fn=lambda role: (
            "Theo 12/2020/NĐ-CP Điều 1, trách nhiệm bao gồm A và B."
            if role == "draft"
            else "Theo 12/2020/NĐ-CP, trách nhiệm gồm A."
        ),
    )
    assert "BUILD_EVIDENCE_PROFILE" in result.stages
    assert "END" in result.stages
    assert result.stages.count("CRITIC_PATCH") <= 1
    assert result.final.critic_calls <= 1
    assert result.final.candidate_count in {1, 2}
    assert result.final.answer_text


def test_ss14_hard_verifier_flags_unsupported_citation() -> None:
    evidence = _packed("Không có mã văn bản trong evidence này.")
    analysis = analyze_requirements("Câu hỏi?")
    profile = build_evidence_profile(evidence)
    draft = attach_draft_attribution(
        "Theo 99/2099/NĐ-CP thì được phép.",
        analysis=analysis,
        evidence_profile=profile,
    )
    verification = verify_draft(
        draft, evidence_text=evidence.rendered_text, evidence_profile=profile
    )
    assert verification.hard_violation is True


def test_ss10_sft_only_baseline_mock() -> None:
    evidence = _packed()
    report = run_sft_only_baseline(
        [("1", "Hỏi?", evidence)],
        generate_fn=lambda q, ev: "Trả lời SFT-only.",
    )
    assert report.status.startswith("pass")
    assert report.cases_per_sec is not None


def test_ss20_to_ss23_artifact_writers(tmp_path: Path) -> None:
    obs = write_observability(tmp_path / "obs.json", GpuObservability(critic_calls=1))
    abl = write_ablation_plan(tmp_path / "abl.json")
    promo = write_promotion_freeze(tmp_path / "promo.json", build_promotion_freeze())
    review = write_adversarial_review(
        tmp_path / "review.json", build_adversarial_review(local_dev=True)
    )
    assert obs.is_file() and abl.is_file() and promo.is_file() and review.is_file()
    plan = build_ablation_plan()
    assert "SFT_only" in plan.arms
    assert build_promotion_freeze().decision == "HOLD"
