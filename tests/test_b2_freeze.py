"""Acceptance tests for the frozen Hybrid-RAG B2 control and drift validator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.finetuned_reader import (
    B2ControlIdentity,
    B2FreezeDriftError,
    B2FreezeIncompleteError,
    build_b2_freeze_fingerprint,
    control_identity_from_config,
    load_b2_freeze_fingerprint,
    require_complete_b2_freeze,
    validate_against_b2_freeze,
)
from legal_rag.finetuned_reader.b2_freeze import UNRESOLVED

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def freeze():
    return load_b2_freeze_fingerprint(REPO_ROOT)


def test_frozen_config_matches_approved_hybrid_rag_hash() -> None:
    live = load_config(REPO_ROOT / "configs" / "hybrid_rag.yaml")
    frozen = load_config(REPO_ROOT / "configs" / "frozen" / "hybrid_rag_b2.yaml")
    assert frozen.config_hash() == live.config_hash()
    assert frozen.retrieval.strategy == "bm25_rerank"
    assert frozen.retrieval.rough_top_n == 12
    assert frozen.evidence.evidence_top_k == 4
    assert frozen.evidence.max_total_chars == 4000
    assert frozen.evidence.max_chunks_per_document == 2


def test_fingerprint_snapshot_matches_builder(freeze) -> None:
    rebuilt = build_b2_freeze_fingerprint(
        REPO_ROOT,
        chunk_cache_fingerprint=freeze.chunk_cache_fingerprint,
        index_fingerprint=freeze.index_fingerprint,
        representative_run_path=freeze.representative_run.get("path"),
    )
    assert freeze.config_hash == rebuilt.config_hash
    assert freeze.prompt_hash == rebuilt.prompt_hash
    assert freeze.evidence_top_k == rebuilt.evidence_top_k
    assert freeze.rough_top_n == rebuilt.rough_top_n
    assert freeze.max_total_chars == rebuilt.max_total_chars
    assert freeze.max_chunks_per_document == rebuilt.max_chunks_per_document
    if freeze.status == "complete":
        assert freeze.chunk_cache_fingerprint != UNRESOLVED
        assert freeze.index_fingerprint != UNRESOLVED
        assert freeze.representative_run["status"] == "recorded"
        assert rebuilt.status == "complete"
    else:
        assert freeze.status == "config_locked_corpus_pending"


def test_complete_freeze_passes_when_snapshot_is_complete(freeze) -> None:
    if freeze.status != "complete":
        pytest.skip("B2 freeze snapshot is not complete in this checkout")
    require_complete_b2_freeze(freeze)
    validate_against_b2_freeze(freeze.control_identity(), freeze)


def test_exact_match_passes(freeze) -> None:
    validate_against_b2_freeze(freeze.control_identity(), freeze)


def test_control_identity_from_frozen_config_passes(freeze) -> None:
    config = load_config(REPO_ROOT / freeze.frozen_config_path)
    candidate = control_identity_from_config(
        config,
        index_fingerprint=freeze.index_fingerprint,
        prompt_hash=freeze.prompt_hash,
    )
    validate_against_b2_freeze(candidate, freeze)


def test_changed_top_k_fails(freeze) -> None:
    candidate = replace(
        freeze.control_identity(),
        evidence_top_k=freeze.evidence_top_k + 1,
    )
    with pytest.raises(B2FreezeDriftError, match="evidence_top_k"):
        validate_against_b2_freeze(candidate, freeze)


def test_changed_rough_top_n_fails(freeze) -> None:
    candidate = replace(freeze.control_identity(), rough_top_n=freeze.rough_top_n + 1)
    with pytest.raises(B2FreezeDriftError, match="rough_top_n"):
        validate_against_b2_freeze(candidate, freeze)


def test_changed_index_fingerprint_fails(freeze) -> None:
    candidate = replace(
        freeze.control_identity(),
        index_fingerprint="drifted-index-fingerprint",
    )
    with pytest.raises(B2FreezeDriftError, match="index_fingerprint"):
        validate_against_b2_freeze(candidate, freeze)


def test_changed_prompt_hash_fails(freeze) -> None:
    candidate = replace(
        freeze.control_identity(),
        prompt_hash="0" * 64,
    )
    with pytest.raises(B2FreezeDriftError, match="prompt_hash"):
        validate_against_b2_freeze(candidate, freeze)


def test_changed_evidence_budget_fails(freeze) -> None:
    candidate = replace(
        freeze.control_identity(),
        max_total_chars=freeze.max_total_chars - 1,
    )
    with pytest.raises(B2FreezeDriftError, match="max_total_chars"):
        validate_against_b2_freeze(candidate, freeze)

    candidate = replace(
        freeze.control_identity(),
        max_chunks_per_document=freeze.max_chunks_per_document + 1,
    )
    with pytest.raises(B2FreezeDriftError, match="max_chunks_per_document"):
        validate_against_b2_freeze(candidate, freeze)


def test_incomplete_freeze_blocks_ftr_build() -> None:
    incomplete = replace(
        load_b2_freeze_fingerprint(REPO_ROOT),
        status="config_locked_corpus_pending",
        chunk_cache_fingerprint=UNRESOLVED,
        index_fingerprint=UNRESOLVED,
        representative_run={
            "status": UNRESOLVED,
            "path": None,
            "reason": "synthetic incomplete fixture",
        },
    )
    with pytest.raises(B2FreezeIncompleteError, match="index_fingerprint"):
        require_complete_b2_freeze(incomplete)


def test_complete_freeze_allows_ftr_when_corpus_resolved() -> None:
    complete = replace(
        load_b2_freeze_fingerprint(REPO_ROOT),
        status="complete",
        context_corpus_status="present",
        context_content_hash="a" * 64,
        chunk_cache_fingerprint="chunk-fp",
        index_fingerprint="index-fp",
        representative_run={
            "status": "recorded",
            "path": "outputs/example_b2_run",
            "reason": None,
        },
    )
    require_complete_b2_freeze(complete)
    validate_against_b2_freeze(
        B2ControlIdentity(
            config_hash=complete.config_hash,
            index_fingerprint="index-fp",
            prompt_hash=complete.prompt_hash,
            rough_top_n=complete.rough_top_n,
            evidence_top_k=complete.evidence_top_k,
            max_total_chars=complete.max_total_chars,
            max_chunks_per_document=complete.max_chunks_per_document,
        ),
        complete,
    )
