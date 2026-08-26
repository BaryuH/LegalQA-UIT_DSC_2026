"""Path-B SEDAR-SFT dataset: LTR rankings → pack evidence → then join gold."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_json
from ..config import ProjectConfig, load_config
from ..finetuned_reader.contracts import EvidenceRecord, ExcludedExample, SFTExample
from ..finetuned_reader.dataset import (
    DatasetBuildError,
    DatasetBuildResult,
    write_dataset_artifacts,
)
from ..finetuned_reader.prompting import GenerativePromptBuilder
from ..finetuned_reader.split_remediation import (
    derive_train_overlap_exclusions,
    exclusion_reason_map,
    normalize_question_text,
)
from ..questions import load_questions
from ..schemas import LegalQuestion
from ..sedar_retrieval.corpus.schema import CanonicalPassage
from ..sedar_retrieval.evidence.passage_packer import (
    PassageEvidenceConfig,
    PassageEvidencePackError,
    RankedPassageCandidate,
    load_retrieval_rankings,
    pack_passage_retrieval_evidence,
)
from ..sedar_retrieval.retrieval.bm25_passages import corpus_fingerprint
from ..sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from .contracts import SEDAR_METHOD, SEDAR_TRAINING_ROLE
from .dataset import SedarDatasetBuildResult, remap_examples_to_sedar_contract

LTR_EVIDENCE_SOURCE = "ltr_passage_rankings"
LTR_DATASET_SCHEMA = "sedar_sft.ss05b.ltr_dataset.v1"


@dataclass(frozen=True, slots=True)
class LtrDatasetBuildConfig:
    """Inputs for an LTR-aligned SFT dataset build."""

    rankings_path: Path
    passages_path: Path
    retrieval_variant: str = "ltr_full_all"
    evidence_top_k: int = 4
    max_total_chars: int = 4000
    max_chunks_per_document: int = 2
    progress_every: int = 100

    def __post_init__(self) -> None:
        if not self.retrieval_variant.strip():
            raise ValueError("retrieval_variant must be non-blank")
        if self.evidence_top_k <= 0:
            raise ValueError("evidence_top_k must be positive")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        if self.max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be positive")
        if self.progress_every <= 0:
            raise ValueError("progress_every must be positive")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def _cross_split_train_exclusions(
    config: ProjectConfig, repo_root: Path
) -> dict[str, str]:
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


def _dataset_output_dir(
    root: Path, settings: Any, *, max_examples: int | None
) -> Path:
    dataset_root = Path(str(settings.output.dataset_root))
    output_dir = root / dataset_root / str(settings.dataset_version)
    if max_examples is not None:
        return output_dir / f"smoke-{max_examples}"
    return output_dir


def build_sft_examples_from_ltr_rankings(
    cases: Sequence[LegalQuestion],
    rankings: Mapping[str, tuple[RankedPassageCandidate, ...]],
    passages: Mapping[str, CanonicalPassage],
    *,
    prompt_builder: GenerativePromptBuilder,
    evidence_config: PassageEvidenceConfig,
    retrieval_config_hash: str,
    index_fingerprint: str,
    overlap_exclusions: Mapping[str, str] | None = None,
    max_examples: int | None = None,
    progress_every: int | None = None,
) -> tuple[
    tuple[SFTExample, ...], tuple[ExcludedExample, ...], tuple[ExcludedExample, ...]
]:
    """Pack LTR evidence first; join gold only after pack+prompt succeed."""

    examples: list[SFTExample] = []
    excluded: list[ExcludedExample] = []
    failures: list[ExcludedExample] = []
    selected_overlap_exclusions = overlap_exclusions or {}
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be greater than zero")
    normalized_questions: dict[str, str] = {}

    for processed_count, case in enumerate(
        sorted(cases, key=lambda item: item.id), start=1
    ):
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

        candidates = rankings.get(case.id)
        if candidates is None:
            failures.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="MISSING_LTR_RANKING",
                    reason="No LTR ranked_ids row for this train case_id",
                )
            )
            continue

        try:
            packed = pack_passage_retrieval_evidence(
                candidates,
                passages,
                config=evidence_config,
            )
            evidence_record = EvidenceRecord.from_packed(
                packed,
                retrieval_config_hash=retrieval_config_hash,
                index_fingerprint=index_fingerprint,
            )
            # Validate train prompt before recording the supervised target.
            prompt_builder.build_training(case.question, packed, case.answer)
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
        except PassageEvidencePackError as exc:
            failures.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="LTR_EVIDENCE_PACK_FAILURE",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue; no silent drop
            failures.append(
                ExcludedExample(
                    case_id=case.id,
                    split="train",
                    reason_code="LTR_PROMPT_OR_PACK_FAILURE",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

        if progress_every is not None and processed_count % progress_every == 0:
            print(
                "LTR_DATASET_BUILD "
                f"processed={processed_count}/{len(cases)} "
                f"examples={len(examples)} excluded={len(excluded)} "
                f"failures={len(failures)}",
                file=sys.stderr,
                flush=True,
            )

    return tuple(examples), tuple(excluded), tuple(failures)


def _write_sedar_overlay(
    output_dir: Path,
    *,
    underlying: DatasetBuildResult,
    sedar_examples: tuple[dict[str, Any], ...],
    build_meta: Mapping[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "sedar_examples.jsonl"
    with examples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sedar_examples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": LTR_DATASET_SCHEMA,
        "method": SEDAR_METHOD,
        "training_role": SEDAR_TRAINING_ROLE,
        "evidence_source": LTR_EVIDENCE_SOURCE,
        "example_count": len(sedar_examples),
        "underlying_dataset_dir": underlying.output_dir.as_posix(),
        "underlying_manifest": (
            (underlying.output_dir / "dataset_manifest.json").as_posix()
            if (underlying.output_dir / "dataset_manifest.json").is_file()
            else None
        ),
        **dict(build_meta),
    }
    manifest_path = output_dir / "sedar_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_sedar_sft_dataset_from_ltr(
    config: ProjectConfig | str | Path,
    *,
    repo_root: str | Path,
    ltr: LtrDatasetBuildConfig,
    max_examples: int | None = None,
) -> SedarDatasetBuildResult:
    """Build effective-train SFT examples from precomputed LTR rankings.

    Contract: rankings must be produced from question-only retrieval. This
    builder never feeds gold into packing; gold joins only after pack succeeds.
    """

    root = Path(repo_root).resolve()
    project = (
        config
        if isinstance(config, ProjectConfig)
        else load_config(root / config if not Path(config).is_absolute() else config)
    )
    if project.project.profile != "sedar_sft":
        raise DatasetBuildError(
            "LTR SEDAR dataset build requires project.profile='sedar_sft'"
        )
    settings = project.finetuned_reader
    if settings is None:
        raise DatasetBuildError("LTR SEDAR dataset build requires finetuned_reader")
    if project.data.split != "train":
        raise DatasetBuildError("SFT dataset construction requires data.split='train'")
    if max_examples is not None and max_examples <= 0:
        raise DatasetBuildError("max_examples must be greater than zero")

    rankings_path = (
        ltr.rankings_path
        if ltr.rankings_path.is_absolute()
        else root / ltr.rankings_path
    )
    passages_path = (
        ltr.passages_path
        if ltr.passages_path.is_absolute()
        else root / ltr.passages_path
    )
    if not rankings_path.is_file():
        raise DatasetBuildError(f"LTR rankings JSONL not found: {rankings_path}")
    if not passages_path.is_file():
        raise DatasetBuildError(f"Passages JSONL not found: {passages_path}")

    exclusions = _cross_split_train_exclusions(project, root)
    if exclusions and settings.overlap_policy == "fail":
        raise DatasetBuildError(
            "Cross-split overlap blocks dataset build; use exclude_and_record"
        )

    prompt_builder = GenerativePromptBuilder.from_files(
        root / settings.train_prompt_path,
        root / settings.inference_prompt_path,
        version=settings.dataset_version,
    )
    cases = load_questions(
        root / project.data.question_path, split="train", include_answers=True
    )
    rankings = load_retrieval_rankings(rankings_path)
    passage_tuple = load_passages_jsonl(str(passages_path))
    passages = {passage.passage_id: passage for passage in passage_tuple}
    index_fingerprint = corpus_fingerprint(passage_tuple)
    evidence_config = PassageEvidenceConfig(
        evidence_top_k=ltr.evidence_top_k,
        max_total_chars=ltr.max_total_chars,
        max_chunks_per_document=ltr.max_chunks_per_document,
    )
    rankings_sha256 = _hash_file(rankings_path)
    passages_sha256 = _hash_file(passages_path)
    retrieval_config_hash = fingerprint_json(
        {
            "evidence_source": LTR_EVIDENCE_SOURCE,
            "retrieval_variant": ltr.retrieval_variant,
            "evidence_top_k": ltr.evidence_top_k,
            "max_total_chars": ltr.max_total_chars,
            "max_chunks_per_document": ltr.max_chunks_per_document,
            "rankings_sha256": rankings_sha256,
            "passages_sha256": passages_sha256,
            "index_fingerprint": index_fingerprint,
        }
    )

    examples, excluded, failures = build_sft_examples_from_ltr_rankings(
        cases,
        rankings,
        passages,
        prompt_builder=prompt_builder,
        evidence_config=evidence_config,
        retrieval_config_hash=retrieval_config_hash,
        index_fingerprint=index_fingerprint,
        overlap_exclusions=exclusions,
        max_examples=max_examples,
        progress_every=ltr.progress_every,
    )
    train_text = _jsonl([example.as_dict() for example in examples])
    manifest: dict[str, object] = {
        "dataset_version": settings.dataset_version,
        "evidence_source": LTR_EVIDENCE_SOURCE,
        "retrieval_variant": ltr.retrieval_variant,
        "overlap_policy": settings.overlap_policy,
        "overlap_remediation_id": settings.overlap_remediation_id,
        "profile": "sedar_sft",
        "source_train_hash": _hash_text(
            (root / project.data.question_path).read_text(encoding="utf-8")
        ),
        "retrieval_config_hash": retrieval_config_hash,
        "index_fingerprint": index_fingerprint,
        "rankings_path": rankings_path.as_posix(),
        "passages_path": passages_path.as_posix(),
        "rankings_sha256": rankings_sha256,
        "passages_sha256": passages_sha256,
        "evidence_packer_hash": fingerprint_json(
            {
                "module": "legal_rag.sedar_retrieval.evidence.passage_packer",
                "version": "task20-v1",
            }
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
        "dataset_scope": "full" if max_examples is None else "smoke",
        "requested_max_examples": max_examples,
        "train_ids_hash": _hash_text(
            "\n".join(example.case_id for example in examples)
        ),
        "example_count": len(examples),
        "excluded_count": len(excluded),
        "retrieval_failure_count": len(failures),
        "examples_hash": _hash_text(train_text),
        "created_at": None,
    }
    output_dir = _dataset_output_dir(root, settings, max_examples=max_examples)
    write_dataset_artifacts(
        output_dir,
        examples=examples,
        excluded=excluded,
        retrieval_failures=failures,
        manifest=manifest,
    )
    underlying = DatasetBuildResult(
        output_dir=output_dir,
        examples=examples,
        excluded=excluded,
        retrieval_failures=failures,
        manifest=manifest,
    )
    sedar_examples = remap_examples_to_sedar_contract(examples)
    overlay_dir = (
        root / settings.output.dataset_root / f"{settings.dataset_version}-sedar"
    )
    if max_examples is not None:
        overlay_dir = overlay_dir / f"smoke-{max_examples}"
    manifest_path = _write_sedar_overlay(
        overlay_dir,
        underlying=underlying,
        sedar_examples=sedar_examples,
        build_meta={
            "retrieval_variant": ltr.retrieval_variant,
            "rankings_sha256": rankings_sha256,
            "passages_sha256": passages_sha256,
        },
    )
    return SedarDatasetBuildResult(
        underlying=underlying,
        sedar_examples=sedar_examples,
        output_dir=overlay_dir,
        manifest_path=manifest_path,
    )


__all__ = [
    "LTR_DATASET_SCHEMA",
    "LTR_EVIDENCE_SOURCE",
    "LtrDatasetBuildConfig",
    "build_sedar_sft_dataset_from_ltr",
    "build_sft_examples_from_ltr_rankings",
]
