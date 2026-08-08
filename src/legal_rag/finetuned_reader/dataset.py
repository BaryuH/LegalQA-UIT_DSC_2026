"""Deterministic, provenance-preserving SFT dataset construction."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_json
from ..config import ProjectConfig
from ..evidence import deduplicate_retrieved_chunks, pack_evidence
from ..generation.prompts import PromptBuilder
from ..pipeline import BM25Preparation, prepare_bm25_index_from_config
from ..questions import load_questions
from ..retrieval import Reranker, create_reranker, retrieve_bm25
from ..schemas import InferenceQuestion, LegalQuestion, PackedEvidence
from .b2_freeze import (
    B2FreezeFingerprint,
    control_identity_from_config,
    load_b2_freeze_fingerprint,
    require_complete_b2_freeze,
    validate_against_b2_freeze,
)
from .contracts import EvidenceRecord, ExcludedExample, SFTExample
from .prompting import GenerativePromptBuilder
from .split_remediation import (
    derive_train_overlap_exclusions,
    exclusion_reason_map,
    normalize_question_text,
)


class DatasetBuildError(RuntimeError):
    """Raised when a dataset cannot be built without a leakage compromise."""


@dataclass(frozen=True, slots=True)
class FrozenRetrievalResult:
    evidence: PackedEvidence
    query_sha256: str
    retrieval_hits: tuple[str, ...]
    reranker: dict[str, object]


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    output_dir: Path
    examples: tuple[SFTExample, ...]
    excluded: tuple[ExcludedExample, ...]
    retrieval_failures: tuple[ExcludedExample, ...]
    manifest: dict[str, object]

    @property
    def status(self) -> str:
        return "pass" if self.examples else "blocked"


class FrozenB2EvidenceRetriever:
    """Use the canonical B2 index, reranker and evidence packer without gold."""

    def __init__(
        self,
        preparation: BM25Preparation,
        freeze: B2FreezeFingerprint,
        reranker: Reranker,
    ) -> None:
        self.preparation = preparation
        self.freeze = freeze
        self.reranker = reranker
        self._chunks = {chunk.chunk_id: chunk for chunk in preparation.chunks}

    @classmethod
    def from_repo(cls, repo_root: str | Path) -> FrozenB2EvidenceRetriever:
        root = Path(repo_root).resolve()
        freeze = load_b2_freeze_fingerprint(root)
        require_complete_b2_freeze(freeze)
        frozen_config_path = root / freeze.frozen_config_path
        preparation = prepare_bm25_index_from_config(
            frozen_config_path,
            repo_root=root,
            rebuild_index=False,
        )
        prompt_builder = PromptBuilder.from_config(
            preparation.config.prompts,
            prompt_dir=root / "configs" / "prompts",
        )
        identity = control_identity_from_config(
            preparation.config,
            index_fingerprint=preparation.index.index_fingerprint,
            prompt_hash=prompt_builder.rag_template.sha256,
        )
        validate_against_b2_freeze(identity, freeze)
        reranker = create_reranker(preparation.config.reranker)
        return cls(preparation, freeze, reranker)

    def retrieve(self, question: InferenceQuestion) -> FrozenRetrievalResult:
        """Retrieve using only the inference-safe question view."""

        if type(question) is not InferenceQuestion:
            raise TypeError("Frozen B2 retrieval accepts InferenceQuestion only")
        query = question.question
        raw_hits = retrieve_bm25(
            self.preparation.index,
            query,
            top_k=self.freeze.rough_top_n,
        )
        if not raw_hits:
            raise DatasetBuildError("No positive BM25 hits for question")
        candidate_texts = {
            hit.chunk_id: self._chunks[hit.chunk_id].retrieval_text
            for hit in raw_hits
            if hit.chunk_id in self._chunks
        }
        reranked = self.reranker.rerank(
            query,
            raw_hits,
            candidate_texts=candidate_texts,
        )
        deduplicated = deduplicate_retrieved_chunks(reranked.hits, self._chunks)
        selected = deduplicated.kept_hits[: self.freeze.evidence_top_k]
        packed = pack_evidence(
            selected,
            self._chunks,
            max_total_chars=self.freeze.max_total_chars,
            max_chunks_per_document=self.freeze.max_chunks_per_document,
            documents=self.preparation.documents,
        )
        return FrozenRetrievalResult(
            evidence=packed,
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            retrieval_hits=tuple(hit.chunk_id for hit in reranked.hits),
            reranker={
                "fallback_reason": reranked.fallback_reason,
                "metadata": dict(reranked.metadata),
                "model": reranked.model,
                "used": reranked.used,
            },
        )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing == content:
                temporary.unlink()
                return
            raise FileExistsError(f"Refusing to overwrite dataset artifact: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_output_dir(root: Path, settings: Any, *, max_examples: int | None) -> Path:
    dataset_root = Path(str(settings.output.dataset_root))
    dataset_version = str(settings.dataset_version)
    output_dir = root / dataset_root / dataset_version
    if max_examples is not None:
        return output_dir / f"smoke-{max_examples}"
    return output_dir


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(f"Cached dataset JSONL is invalid: {path}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise DatasetBuildError(f"Cached dataset JSONL must contain objects: {path}")
    return records


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise DatasetBuildError(f"Cached dataset field must be a string: {field_name}")
    return value


def _string_tuple_field(
    payload: Mapping[str, object], field_name: str
) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DatasetBuildError(
            f"Cached dataset field must be a list of strings: {field_name}"
        )
    return tuple(value)


def _sft_example_from_dict(payload: Mapping[str, object]) -> SFTExample:
    evidence_payload = payload.get("evidence")
    if not isinstance(evidence_payload, dict):
        raise DatasetBuildError("Cached SFT example must contain evidence object")
    evidence = EvidenceRecord(
        rendered_text=_string_field(evidence_payload, "rendered_text"),
        chunk_ids=_string_tuple_field(evidence_payload, "chunk_ids"),
        document_ids=_string_tuple_field(evidence_payload, "document_ids"),
        retrieval_config_hash=_string_field(evidence_payload, "retrieval_config_hash"),
        index_fingerprint=_string_field(evidence_payload, "index_fingerprint"),
        packed_evidence_hash=_string_field(evidence_payload, "packed_evidence_hash"),
    )
    try:
        return SFTExample(
            example_id=_string_field(payload, "example_id"),
            case_id=_string_field(payload, "case_id"),
            question=_string_field(payload, "question"),
            evidence=evidence,
            target_answer=_string_field(payload, "target_answer"),
            split=_string_field(payload, "split"),
        )
    except ValueError as exc:
        raise DatasetBuildError(
            "Cached SFT example violates the training contract"
        ) from exc


def _excluded_example_from_dict(payload: Mapping[str, object]) -> ExcludedExample:
    return ExcludedExample(
        case_id=_string_field(payload, "case_id"),
        split=_string_field(payload, "split"),
        reason_code=_string_field(payload, "reason_code"),
        reason=_string_field(payload, "reason"),
    )


def _load_cached_dataset(
    output_dir: Path,
    *,
    cache_identity: Mapping[str, object],
    max_examples: int | None,
) -> DatasetBuildResult | None:
    """Load a complete, identity-matching dataset artifact without retrieval work."""

    required_paths = (
        output_dir / "train.jsonl",
        output_dir / "excluded.jsonl",
        output_dir / "retrieval_failures.jsonl",
        output_dir / "dataset_manifest.json",
        output_dir / "statistics.json",
    )
    present_paths = [path.exists() for path in required_paths]
    if not any(present_paths):
        return None
    if not all(present_paths):
        raise DatasetBuildError(f"Cached dataset is incomplete: {output_dir}")
    try:
        manifest = json.loads(
            (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        statistics = json.loads(
            (output_dir / "statistics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(
            f"Cached dataset metadata is invalid: {output_dir}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(statistics, dict):
        raise DatasetBuildError(
            f"Cached dataset metadata must be objects: {output_dir}"
        )
    if any(
        manifest.get(field_name) != value
        for field_name, value in cache_identity.items()
    ):
        return None
    cached_scope = manifest.get("dataset_scope", "full")
    if cached_scope != ("full" if max_examples is None else "smoke"):
        return None
    if manifest.get("requested_max_examples") != max_examples:
        return None

    examples = tuple(
        _sft_example_from_dict(record)
        for record in _read_jsonl_records(output_dir / "train.jsonl")
    )
    excluded = tuple(
        _excluded_example_from_dict(record)
        for record in _read_jsonl_records(output_dir / "excluded.jsonl")
    )
    failures = tuple(
        _excluded_example_from_dict(record)
        for record in _read_jsonl_records(output_dir / "retrieval_failures.jsonl")
    )
    train_text = _jsonl([example.as_dict() for example in examples])
    if (
        manifest.get("examples_hash") != _hash_text(train_text)
        or manifest.get("example_count") != len(examples)
        or manifest.get("excluded_count") != len(excluded)
        or manifest.get("retrieval_failure_count") != len(failures)
        or statistics.get("train_jsonl_sha256") != _hash_text(train_text)
    ):
        raise DatasetBuildError(f"Cached dataset integrity check failed: {output_dir}")
    return DatasetBuildResult(
        output_dir=output_dir,
        examples=examples,
        excluded=excluded,
        retrieval_failures=failures,
        manifest=manifest,
    )


def build_sft_dataset(
    cases: Sequence[LegalQuestion],
    *,
    retriever: FrozenB2EvidenceRetriever,
    prompt_builder: GenerativePromptBuilder,
    retrieval_config_hash: str,
    overlap_exclusions: Mapping[str, str] | None = None,
    max_examples: int | None = None,
) -> tuple[
    tuple[SFTExample, ...], tuple[ExcludedExample, ...], tuple[ExcludedExample, ...]
]:
    """Build stable examples after retrieval; all failures are recorded."""

    examples: list[SFTExample] = []
    excluded: list[ExcludedExample] = []
    failures: list[ExcludedExample] = []
    selected_overlap_exclusions = overlap_exclusions or {}
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be greater than zero")
    normalized_questions: dict[str, str] = {}
    for case in sorted(cases, key=lambda item: item.id):
        if case.id in selected_overlap_exclusions:
            excluded.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="CROSS_SPLIT_OVERLAP",
                    reason=selected_overlap_exclusions[case.id],
                )
            )
            continue
        if case.answer is None or not case.answer.strip():
            excluded.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="BLANK_TARGET",
                    reason="Train target is missing or blank",
                )
            )
            continue
        normalized = normalize_question_text(case.question)
        if normalized in normalized_questions:
            excluded.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="DUPLICATE_NORMALIZED_QUESTION",
                    reason=(
                        f"Duplicate of train case {normalized_questions[normalized]}"
                    ),
                )
            )
            continue
        normalized_questions[normalized] = case.id
        try:
            retrieved = retriever.retrieve(case.inference_view())
            evidence_record = EvidenceRecord.from_packed(
                retrieved.evidence,
                retrieval_config_hash=retrieval_config_hash,
                index_fingerprint=retriever.preparation.index.index_fingerprint,
            )
            # Render validation is intentionally before target attachment to preserve
            # the question/evidence-only inference boundary.
            prompt_builder.build_training(
                case.question, retrieved.evidence, case.answer
            )
            examples.append(
                SFTExample(
                    example_id=f"train::{case.id}",
                    case_id=case.id,
                    question=case.question,
                    evidence=evidence_record,
                    target_answer=case.answer,
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
        except Exception as exc:
            failures.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="RETRIEVAL_OR_PROMPT_FAILURE",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(examples), tuple(excluded), tuple(failures)


def write_dataset_artifacts(
    output_dir: str | Path,
    *,
    examples: Sequence[SFTExample],
    excluded: Sequence[ExcludedExample],
    retrieval_failures: Sequence[ExcludedExample],
    manifest: Mapping[str, object],
) -> Path:
    """Write the complete dataset tree atomically and deterministically."""

    root = Path(output_dir)
    train_text = _jsonl([example.as_dict() for example in examples])
    validation_text = ""
    excluded_text = _jsonl([item.as_dict() for item in excluded])
    failures_text = _jsonl([item.as_dict() for item in retrieval_failures])
    _write_atomic(root / "train.jsonl", train_text)
    _write_atomic(root / "validation.jsonl", validation_text)
    _write_atomic(root / "excluded.jsonl", excluded_text)
    _write_atomic(root / "retrieval_failures.jsonl", failures_text)
    _write_atomic(
        root / "dataset_manifest.json",
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    stats = {
        "excluded_count": len(excluded),
        "example_count": len(examples),
        "retrieval_failure_count": len(retrieval_failures),
        "train_jsonl_sha256": _hash_text(train_text),
        "validation_jsonl_sha256": _hash_text(validation_text),
    }
    _write_atomic(
        root / "statistics.json",
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return root


def _cross_split_train_exclusions(
    config: ProjectConfig, repo_root: Path
) -> dict[str, str]:
    """Return configured train exclusions using the shared Unicode-safe policy."""

    train = load_questions(
        repo_root / config.data.question_path, split="train", include_answers=True
    )
    data_dir = repo_root / config.data.data_dir
    comparison_questions: dict[str, dict[str, str]] = {}
    for split, filename in (
        ("warmup", "warmup.json"),
        ("public", "public-official.json"),
        ("private", "private-official.json"),
    ):
        path = data_dir / filename
        if path.is_file():
            comparison_questions[split] = {
                case.id: case.question
                for case in load_questions(path, split=split, include_answers=False)
            }
    exclusions = derive_train_overlap_exclusions(
        {case.id: case.question for case in train}, comparison_questions
    )
    return exclusion_reason_map(exclusions)


def build_sft_dataset_from_config(
    config: ProjectConfig,
    *,
    repo_root: str | Path,
    max_examples: int | None = None,
) -> DatasetBuildResult:
    """Build the configured train-only dataset, failing closed on overlap policy."""

    settings = config.finetuned_reader
    if settings is None or not settings.enabled:
        raise DatasetBuildError("Generative finetuned_reader settings are required")
    if config.data.split != "train":
        raise DatasetBuildError("SFT dataset construction requires data.split='train'")
    if max_examples is not None and max_examples <= 0:
        raise DatasetBuildError("max_examples must be greater than zero")
    root = Path(repo_root).resolve()
    exclusions = _cross_split_train_exclusions(config, root)
    if exclusions and settings.overlap_policy == "fail":
        raise DatasetBuildError(
            "Cross-split overlap blocks dataset build; use an approved "
            "exclude_and_record policy only with a documented decision"
        )
    retriever = FrozenB2EvidenceRetriever.from_repo(root)
    prompt_builder = GenerativePromptBuilder.from_files(
        root / settings.train_prompt_path,
        root / settings.inference_prompt_path,
        version=settings.dataset_version,
    )
    cases = load_questions(
        root / config.data.question_path, split="train", include_answers=True
    )
    cache_identity = {
        "dataset_version": settings.dataset_version,
        "overlap_policy": settings.overlap_policy,
        "overlap_remediation_id": settings.overlap_remediation_id,
        "profile": "finetuned_reader",
        "source_train_hash": _hash_text(
            (root / config.data.question_path).read_text(encoding="utf-8")
        ),
        "source_validation_hash": None,
        "retrieval_config_hash": retriever.freeze.config_hash,
        "index_fingerprint": retriever.preparation.index.index_fingerprint,
        "evidence_packer_hash": fingerprint_json(
            {"module": "legal_rag.evidence", "version": "v1"}
        ),
        "prompt_version": settings.dataset_version,
        "prompt_hash": prompt_builder.inference_sha256,
        "cross_split_exclusions_hash": _hash_text(
            _jsonl(
                [
                    {"case_id": case_id, "reason": reason}
                    for case_id, reason in sorted(exclusions.items())
                ]
            )
        ),
    }
    output_dir = _dataset_output_dir(root, settings, max_examples=max_examples)
    cached = _load_cached_dataset(
        output_dir,
        cache_identity=cache_identity,
        max_examples=max_examples,
    )
    if cached is not None:
        return cached
    examples, excluded, failures = build_sft_dataset(
        cases,
        retriever=retriever,
        prompt_builder=prompt_builder,
        retrieval_config_hash=retriever.freeze.config_hash,
        overlap_exclusions=exclusions,
        max_examples=max_examples,
    )
    train_text = _jsonl([example.as_dict() for example in examples])
    manifest: dict[str, object] = {
        **cache_identity,
        "dataset_scope": "full" if max_examples is None else "smoke",
        "requested_max_examples": max_examples,
        "train_ids_hash": _hash_text(
            "\n".join(example.case_id for example in examples)
        ),
        "validation_ids_hash": None,
        "example_count": len(examples),
        "excluded_count": len(excluded),
        "retrieval_failure_count": len(failures),
        "examples_hash": _hash_text(train_text),
        "created_at": None,
    }
    write_dataset_artifacts(
        output_dir,
        examples=examples,
        excluded=excluded,
        retrieval_failures=failures,
        manifest=manifest,
    )
    return DatasetBuildResult(
        output_dir=output_dir,
        examples=examples,
        excluded=excluded,
        retrieval_failures=failures,
        manifest=manifest,
    )


__all__ = [
    "DatasetBuildError",
    "DatasetBuildResult",
    "FrozenB2EvidenceRetriever",
    "FrozenRetrievalResult",
    "build_sft_dataset",
    "build_sft_dataset_from_config",
    "write_dataset_artifacts",
]
