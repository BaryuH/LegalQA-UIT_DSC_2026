"""Frozen Hybrid-RAG (B2) control identity and drift validation for FTR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import ProjectConfig, load_config
from ..evaluation.evaluator import (
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    METRIC_CONTRACT_VERSION,
)
from ..generation.prompts import load_prompt_template
from ..retrieval.bm25 import BM25_INDEX_SCHEMA_VERSION
from ..text.cache import CACHE_SCHEMA_VERSION
from ..text.normalize import NORMALIZATION_VERSION

FREEZE_SCHEMA_VERSION = "ftr02.b2_freeze.v1"
FREEZE_ID = "hybrid_rag_b2"
DEFAULT_FROZEN_CONFIG_REL = Path("configs/frozen/hybrid_rag_b2.yaml")
DEFAULT_FINGERPRINT_REL = Path("artifacts/b2_freeze/fingerprint.json")
SOURCE_CONFIG_REL = Path("configs/hybrid_rag.yaml")
RAG_PROMPT_REL = Path("configs/prompts/rag_v1.txt")
UNRESOLVED = "UNRESOLVED"


class B2FreezeDriftError(ValueError):
    """Raised when a candidate B2 control drifts from the frozen snapshot."""


class B2FreezeIncompleteError(ValueError):
    """Raised when corpus-dependent freeze fields are still unresolved."""


@dataclass(frozen=True, slots=True)
class B2ControlIdentity:
    """Observed B2 control values used when building or evaluating FTR examples."""

    config_hash: str
    index_fingerprint: str
    prompt_hash: str
    rough_top_n: int
    evidence_top_k: int
    max_total_chars: int
    max_chunks_per_document: int


@dataclass(frozen=True, slots=True)
class B2FreezeFingerprint:
    """Immutable B2 freeze snapshot used as the FTR comparison control."""

    schema_version: str
    freeze_id: str
    status: str
    source_config_path: str
    frozen_config_path: str
    config_hash: str
    data_manifest_hash: str
    context_corpus_status: str
    context_content_hash: str
    chunk_cache_fingerprint: str
    index_fingerprint: str
    chunking: dict[str, Any]
    retrieval: dict[str, Any]
    reranker: dict[str, Any]
    evidence: dict[str, Any]
    prompt: dict[str, str]
    generation: dict[str, Any]
    evaluator: dict[str, str]
    normalization_version: str
    chunk_cache_schema_version: str
    bm25_index_schema_version: str
    representative_run: dict[str, Any]

    @property
    def prompt_hash(self) -> str:
        return self.prompt["sha256"]

    @property
    def evidence_top_k(self) -> int:
        return int(self.evidence["evidence_top_k"])

    @property
    def rough_top_n(self) -> int:
        return int(self.retrieval["rough_top_n"])

    @property
    def max_total_chars(self) -> int:
        return int(self.evidence["max_total_chars"])

    @property
    def max_chunks_per_document(self) -> int:
        return int(self.evidence["max_chunks_per_document"])

    def control_identity(self) -> B2ControlIdentity:
        return B2ControlIdentity(
            config_hash=self.config_hash,
            index_fingerprint=self.index_fingerprint,
            prompt_hash=self.prompt_hash,
            rough_top_n=self.rough_top_n,
            evidence_top_k=self.evidence_top_k,
            max_total_chars=self.max_total_chars,
            max_chunks_per_document=self.max_chunks_per_document,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> B2FreezeFingerprint:
        required = {
            "schema_version",
            "freeze_id",
            "status",
            "source_config_path",
            "frozen_config_path",
            "config_hash",
            "data_manifest_hash",
            "context_corpus_status",
            "context_content_hash",
            "chunk_cache_fingerprint",
            "index_fingerprint",
            "chunking",
            "retrieval",
            "reranker",
            "evidence",
            "prompt",
            "generation",
            "evaluator",
            "normalization_version",
            "chunk_cache_schema_version",
            "bm25_index_schema_version",
            "representative_run",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"B2 freeze fingerprint missing fields: {sorted(missing)}")
        return cls(
            schema_version=str(payload["schema_version"]),
            freeze_id=str(payload["freeze_id"]),
            status=str(payload["status"]),
            source_config_path=str(payload["source_config_path"]),
            frozen_config_path=str(payload["frozen_config_path"]),
            config_hash=str(payload["config_hash"]),
            data_manifest_hash=str(payload["data_manifest_hash"]),
            context_corpus_status=str(payload["context_corpus_status"]),
            context_content_hash=str(payload["context_content_hash"]),
            chunk_cache_fingerprint=str(payload["chunk_cache_fingerprint"]),
            index_fingerprint=str(payload["index_fingerprint"]),
            chunking=dict(payload["chunking"]),  # type: ignore[arg-type]
            retrieval=dict(payload["retrieval"]),  # type: ignore[arg-type]
            reranker=dict(payload["reranker"]),  # type: ignore[arg-type]
            evidence=dict(payload["evidence"]),  # type: ignore[arg-type]
            prompt={str(k): str(v) for k, v in dict(payload["prompt"]).items()},
            generation=dict(payload["generation"]),  # type: ignore[arg-type]
            evaluator={str(k): str(v) for k, v in dict(payload["evaluator"]).items()},
            normalization_version=str(payload["normalization_version"]),
            chunk_cache_schema_version=str(payload["chunk_cache_schema_version"]),
            bm25_index_schema_version=str(payload["bm25_index_schema_version"]),
            representative_run=dict(payload["representative_run"]),  # type: ignore[arg-type]
        )


def _json_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _data_manifest_hash(repo_root: Path) -> str:
    from scripts.verify_data_manifest import (
        DEFAULT_MANIFEST_PATH,
        build_manifest,
        verify_manifest,
    )

    root = repo_root.resolve()
    manifest = build_manifest(root)
    verify_manifest(root, root / DEFAULT_MANIFEST_PATH)
    return _json_hash(manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def control_identity_from_config(
    config: ProjectConfig,
    *,
    index_fingerprint: str,
    prompt_hash: str,
) -> B2ControlIdentity:
    """Build a comparable control identity from a live Hybrid-RAG config."""

    return B2ControlIdentity(
        config_hash=config.config_hash(),
        index_fingerprint=index_fingerprint,
        prompt_hash=prompt_hash,
        rough_top_n=config.retrieval.rough_top_n,
        evidence_top_k=config.evidence.evidence_top_k,
        max_total_chars=config.evidence.max_total_chars,
        max_chunks_per_document=config.evidence.max_chunks_per_document,
    )


def build_b2_freeze_fingerprint(
    repo_root: str | Path,
    *,
    frozen_config_path: str | Path | None = None,
    chunk_cache_fingerprint: str = UNRESOLVED,
    index_fingerprint: str = UNRESOLVED,
    representative_run_path: str | None = None,
) -> B2FreezeFingerprint:
    """Resolve the approved B2 freeze snapshot from repository files."""

    root = Path(repo_root).resolve()
    config_rel = (
        Path(frozen_config_path)
        if frozen_config_path is not None
        else DEFAULT_FROZEN_CONFIG_REL
    )
    config_path = (root / config_rel).resolve()
    config = load_config(config_path)
    if config.retrieval.strategy != "bm25_rerank":
        raise ValueError(
            "Frozen B2 config must use retrieval.strategy=bm25_rerank; "
            f"got {config.retrieval.strategy!r}"
        )

    source_config = load_config(root / SOURCE_CONFIG_REL)
    if config.config_hash() != source_config.config_hash():
        raise B2FreezeDriftError(
            "Frozen B2 config hash drifted from configs/hybrid_rag.yaml; "
            "refuse to snapshot a tuned profile"
        )

    prompt_path = root / RAG_PROMPT_REL
    prompt = load_prompt_template(
        prompt_path,
        name="rag",
        version=config.prompts.rag_version,
    )
    contexts_path = (root / config.data.selected_contexts_path).resolve()
    if contexts_path.is_file():
        context_status = "present"
        context_hash = _sha256_file(contexts_path)
    else:
        context_status = "missing"
        context_hash = UNRESOLVED

    corpus_resolved = (
        context_status == "present"
        and chunk_cache_fingerprint != UNRESOLVED
        and index_fingerprint != UNRESOLVED
        and bool(representative_run_path)
    )
    status = "complete" if corpus_resolved else "config_locked_corpus_pending"

    representative_run: dict[str, Any]
    if representative_run_path:
        representative_run = {
            "status": "recorded",
            "path": representative_run_path,
            "reason": None,
        }
    else:
        representative_run = {
            "status": UNRESOLVED,
            "path": None,
            "reason": (
                "selected-contexts.zip missing or no reproducible Hybrid-RAG "
                "warmup run recorded for competition corpus"
                if context_status == "missing"
                else "representative Hybrid-RAG run artifact not yet recorded"
            ),
        }

    return B2FreezeFingerprint(
        schema_version=FREEZE_SCHEMA_VERSION,
        freeze_id=FREEZE_ID,
        status=status,
        source_config_path=SOURCE_CONFIG_REL.as_posix(),
        frozen_config_path=config_rel.as_posix(),
        config_hash=config.config_hash(),
        data_manifest_hash=_data_manifest_hash(root),
        context_corpus_status=context_status,
        context_content_hash=context_hash,
        chunk_cache_fingerprint=chunk_cache_fingerprint,
        index_fingerprint=index_fingerprint,
        chunking={
            "max_chars": config.chunking.max_chars,
            "overlap_chars": config.chunking.overlap_chars,
            "min_chars": config.chunking.min_chars,
            "version": config.chunking.version,
        },
        retrieval={
            "strategy": config.retrieval.strategy,
            "rough_top_n": config.retrieval.rough_top_n,
            "k1": config.retrieval.k1,
            "b": config.retrieval.b,
        },
        reranker={
            "enabled": config.reranker.enabled,
            "required": config.reranker.required,
            "provider": config.reranker.provider,
            "model": config.reranker.model,
            "model_revision": config.reranker.model_revision,
            "device": config.reranker.device,
            "batch_size": config.reranker.batch_size,
            "max_length": config.reranker.max_length,
        },
        evidence={
            "evidence_top_k": config.evidence.evidence_top_k,
            "max_total_chars": config.evidence.max_total_chars,
            "max_chunks_per_document": config.evidence.max_chunks_per_document,
        },
        prompt={
            "name": prompt.name,
            "version": prompt.version,
            "path": RAG_PROMPT_REL.as_posix(),
            "sha256": prompt.sha256,
        },
        generation={
            "provider": config.generation.provider,
            "model": config.generation.model,
            "temperature": config.generation.temperature,
            "max_output_chars": config.generation.max_output_chars,
            "max_completion_length": config.generation.max_completion_length,
            "retries": config.generation.retries,
        },
        evaluator={
            "name": EVALUATOR_NAME,
            "version": EVALUATOR_VERSION,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        },
        normalization_version=NORMALIZATION_VERSION,
        chunk_cache_schema_version=CACHE_SCHEMA_VERSION,
        bm25_index_schema_version=BM25_INDEX_SCHEMA_VERSION,
        representative_run=representative_run,
    )


def load_b2_freeze_fingerprint(
    repo_root: str | Path,
    *,
    fingerprint_path: str | Path | None = None,
) -> B2FreezeFingerprint:
    """Load the committed B2 freeze fingerprint from disk."""

    root = Path(repo_root).resolve()
    path = root / (
        Path(fingerprint_path)
        if fingerprint_path is not None
        else DEFAULT_FINGERPRINT_REL
    )
    if not path.is_file():
        raise FileNotFoundError(f"B2 freeze fingerprint not found: {path}")
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("B2 freeze fingerprint must be a JSON object")
    return B2FreezeFingerprint.from_dict(raw)


def write_b2_freeze_fingerprint(
    repo_root: str | Path,
    fingerprint: B2FreezeFingerprint,
    *,
    fingerprint_path: str | Path | None = None,
) -> Path:
    """Write fingerprint JSON with stable key ordering."""

    root = Path(repo_root).resolve()
    path = root / (
        Path(fingerprint_path)
        if fingerprint_path is not None
        else DEFAULT_FINGERPRINT_REL
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fingerprint.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def require_complete_b2_freeze(freeze: B2FreezeFingerprint) -> None:
    """Fail closed when corpus/index fingerprints are not yet frozen."""

    unresolved: list[str] = []
    if freeze.context_corpus_status != "present":
        unresolved.append("context_corpus")
    if freeze.context_content_hash == UNRESOLVED:
        unresolved.append("context_content_hash")
    if freeze.chunk_cache_fingerprint == UNRESOLVED:
        unresolved.append("chunk_cache_fingerprint")
    if freeze.index_fingerprint == UNRESOLVED:
        unresolved.append("index_fingerprint")
    if freeze.representative_run.get("status") == UNRESOLVED:
        unresolved.append("representative_run")
    if unresolved:
        raise B2FreezeIncompleteError(
            "B2 freeze is incomplete for finetuned_reader build/evaluate: "
            + ", ".join(unresolved)
        )


def validate_against_b2_freeze(
    candidate: B2ControlIdentity,
    freeze: B2FreezeFingerprint,
) -> None:
    """Fail closed when candidate B2 controls drift from the frozen snapshot."""

    expected = freeze.control_identity()
    mismatches: list[str] = []
    for field_name in (
        "config_hash",
        "index_fingerprint",
        "prompt_hash",
        "rough_top_n",
        "evidence_top_k",
        "max_total_chars",
        "max_chunks_per_document",
    ):
        observed = getattr(candidate, field_name)
        required = getattr(expected, field_name)
        if observed != required:
            mismatches.append(f"{field_name}: expected {required!r}, got {observed!r}")
    if mismatches:
        raise B2FreezeDriftError(
            "B2 freeze drift detected for finetuned_reader: " + "; ".join(mismatches)
        )


__all__ = [
    "B2ControlIdentity",
    "B2FreezeDriftError",
    "B2FreezeFingerprint",
    "B2FreezeIncompleteError",
    "DEFAULT_FINGERPRINT_REL",
    "DEFAULT_FROZEN_CONFIG_REL",
    "FREEZE_ID",
    "FREEZE_SCHEMA_VERSION",
    "UNRESOLVED",
    "build_b2_freeze_fingerprint",
    "control_identity_from_config",
    "load_b2_freeze_fingerprint",
    "require_complete_b2_freeze",
    "validate_against_b2_freeze",
    "write_b2_freeze_fingerprint",
]
