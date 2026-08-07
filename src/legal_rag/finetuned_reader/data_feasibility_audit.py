"""Read-only LegalQACompetition fine-tuning data feasibility and leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..config import ProjectConfig, load_config
from ..evidence import deduplicate_retrieved_chunks, pack_evidence
from ..generation.prompts import PromptBuilder
from ..questions import QuestionLoadError, inspect_questions, load_questions
from ..retrieval import BM25Index, retrieve_bm25
from ..schemas import LegalChunk, LegalDocument, LegalQuestion, PackedEvidence
from ..splits import SPLIT_USAGE_REGISTRY, SplitName
from ..text.normalize import tokenize_legal_text
from .b2_freeze import (
    DEFAULT_FROZEN_CONFIG_REL,
    UNRESOLVED,
    load_b2_freeze_fingerprint,
    require_complete_b2_freeze,
)
from .split_remediation import (
    TrainingOverlapExclusion,
    derive_train_overlap_exclusions,
    normalize_question_text,
)

AUDIT_SCHEMA_VERSION = "ftr03.data-feasibility.v1"
DEFAULT_AUDIT_DIR = Path("artifacts/finetuned_reader_audit")
DEFAULT_SPLIT_PATHS: dict[SplitName, Path] = {
    "train": Path("data/train.json"),
    "warmup": Path("data/warmup.json"),
    "public": Path("data/public-official.json"),
    "private": Path("data/private-official.json"),
}
TRAINING_ALLOWED_SPLITS: frozenset[SplitName] = frozenset({"train"})
# Local diagnostic only until FTR-04 locks a tokenizer and max_seq_length.
PROVISIONAL_MAX_SEQ_TOKENS = 4096
LOW_OVERLAP_THRESHOLD = 0.05
_FORBIDDEN_INDEX_KEYS = frozenset(
    {
        "answer",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
        "label",
    }
)


class DataFeasibilityAuditError(ValueError):
    """Raised when the audit cannot proceed on invalid fixture inputs."""


@dataclass(frozen=True, slots=True)
class LengthStats:
    count: int
    min: int | None
    max: int | None
    mean: float | None
    p50: float | None
    p90: float | None
    p95: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    split: SplitName
    path: str
    status: Literal["present", "missing", "error"]
    error: str | None
    record_count: int
    unique_id_count: int
    duplicate_id_count: int
    blank_question_count: int
    blank_answer_count: int
    missing_answer_count: int
    question_type: str
    answer_type: str
    question_length_chars: LengthStats
    answer_length_chars: LengthStats
    question_length_tokens: LengthStats
    answer_length_tokens: LengthStats
    exact_duplicate_questions: int
    normalized_duplicate_questions: int
    duplicate_qa_pairs: int
    ids: tuple[str, ...] = ()
    questions_by_id: Mapping[str, str] = field(default_factory=dict)
    answers_by_id: Mapping[str, str | None] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Serialize without embedding question/answer text."""

        return {
            "split": self.split,
            "path": self.path,
            "status": self.status,
            "error": self.error,
            "record_count": self.record_count,
            "unique_id_count": self.unique_id_count,
            "duplicate_id_count": self.duplicate_id_count,
            "blank_question_count": self.blank_question_count,
            "blank_answer_count": self.blank_answer_count,
            "missing_answer_count": self.missing_answer_count,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "question_length_chars": self.question_length_chars.as_dict(),
            "answer_length_chars": self.answer_length_chars.as_dict(),
            "question_length_tokens": self.question_length_tokens.as_dict(),
            "answer_length_tokens": self.answer_length_tokens.as_dict(),
            "exact_duplicate_questions": self.exact_duplicate_questions,
            "normalized_duplicate_questions": self.normalized_duplicate_questions,
            "duplicate_qa_pairs": self.duplicate_qa_pairs,
        }


@dataclass(frozen=True, slots=True)
class CaseRetrievalAudit:
    id: str
    split: SplitName
    query: str
    status: str
    top1_exact_substring: bool | None
    top3_exact_substring: bool | None
    topk_exact_substring: bool | None
    top1_normalized_substring: bool | None
    top3_normalized_substring: bool | None
    topk_normalized_substring: bool | None
    answer_evidence_token_overlap: float | None
    zero_evidence: bool
    low_overlap: bool | None
    retrieved_document_ids: tuple[str, ...]
    evidence_token_count: int | None
    prompt_token_count: int | None
    target_token_count: int | None
    prompt_plus_target_tokens: int | None
    fits_provisional_max_seq: bool | None
    fit_status: str
    gold_in_query: bool
    gold_in_evidence: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep per-case artifact free of gold/question text for safety.
        payload.pop("query", None)
        return payload


@dataclass(frozen=True, slots=True)
class AuditResult:
    schema_version: str
    status: str
    hard_stop: bool
    hard_stop_reasons: tuple[str, ...]
    b2_freeze: dict[str, Any]
    splits: dict[str, dict[str, Any]]
    cross_split: dict[str, Any]
    training_remediation: dict[str, Any]
    training_exclusions: tuple[TrainingOverlapExclusion, ...]
    retrieval: dict[str, Any]
    fit: dict[str, Any]
    leakage: dict[str, Any]
    source_integrity: dict[str, Any]
    per_case: tuple[CaseRetrievalAudit, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "hard_stop": self.hard_stop,
            "hard_stop_reasons": list(self.hard_stop_reasons),
            "b2_freeze": self.b2_freeze,
            "splits": self.splits,
            "cross_split": self.cross_split,
            "training_remediation": self.training_remediation,
            "retrieval": self.retrieval,
            "fit": self.fit,
            "leakage": self.leakage,
            "source_integrity": self.source_integrity,
            "per_case_count": len(self.per_case),
        }


def _percentile(sorted_values: Sequence[int], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def length_stats(values: Sequence[int]) -> LengthStats:
    if not values:
        return LengthStats(0, None, None, None, None, None, None)
    ordered = tuple(sorted(values))
    return LengthStats(
        count=len(ordered),
        min=ordered[0],
        max=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
    )


def _empty_split(
    split: SplitName, path: Path, status: str, error: str | None
) -> SplitAudit:
    empty = length_stats(())
    return SplitAudit(
        split=split,
        path=path.as_posix(),
        status=status,  # type: ignore[arg-type]
        error=error,
        record_count=0,
        unique_id_count=0,
        duplicate_id_count=0,
        blank_question_count=0,
        blank_answer_count=0,
        missing_answer_count=0,
        question_type="absent",
        answer_type="absent",
        question_length_chars=empty,
        answer_length_chars=empty,
        question_length_tokens=empty,
        answer_length_tokens=empty,
        exact_duplicate_questions=0,
        normalized_duplicate_questions=0,
        duplicate_qa_pairs=0,
    )


def audit_split_file(
    path: Path,
    *,
    split: SplitName,
    include_answers: bool,
) -> SplitAudit:
    """Audit one question file without modifying it."""

    if not path.is_file():
        return _empty_split(split, path, "missing", f"Source file missing: {path}")

    try:
        stats = inspect_questions(path)
    except QuestionLoadError as exc:
        return _empty_split(split, path, "error", str(exc))

    try:
        # For public/private, never materialize answers even if present on disk.
        records = load_questions(path, split=split, include_answers=include_answers)
    except QuestionLoadError as exc:
        empty = length_stats(())
        return SplitAudit(
            split=split,
            path=path.as_posix(),
            status="error",
            error=str(exc),
            record_count=stats.record_count,
            unique_id_count=stats.record_count - stats.duplicate_id_count,
            duplicate_id_count=stats.duplicate_id_count,
            blank_question_count=stats.blank_question_count,
            blank_answer_count=stats.blank_answer_count,
            missing_answer_count=stats.answer_missing_count,
            question_type="str",
            answer_type="str" if stats.answer_available_count else "missing",
            question_length_chars=length_stats(stats.question_lengths),
            answer_length_chars=length_stats(stats.answer_lengths),
            question_length_tokens=empty,
            answer_length_tokens=empty,
            exact_duplicate_questions=0,
            normalized_duplicate_questions=0,
            duplicate_qa_pairs=0,
        )

    ids = [record.id for record in records]
    unique_ids = set(ids)
    questions = [record.question for record in records]
    answers = [record.answer for record in records]

    blank_questions = sum(1 for text in questions if not text.strip())
    blank_answers = 0
    missing_answers = 0
    answer_lengths_chars: list[int] = []
    answer_lengths_tokens: list[int] = []
    for answer in answers:
        if answer is None:
            missing_answers += 1
            continue
        if not answer.strip():
            blank_answers += 1
        answer_lengths_chars.append(len(answer))
        answer_lengths_tokens.append(len(tokenize_legal_text(answer)))

    exact_counts = Counter(questions)
    exact_dupes = sum(1 for count in exact_counts.values() if count > 1)
    normalized_counts = Counter(normalize_question_text(text) for text in questions)
    normalized_dupes = sum(1 for count in normalized_counts.values() if count > 1)

    qa_pairs = [
        (normalize_question_text(record.question), (record.answer or "").strip())
        for record in records
        if record.answer is not None
    ]
    qa_counts = Counter(qa_pairs)
    duplicate_qa = sum(1 for count in qa_counts.values() if count > 1)

    question_types = {"str"} if questions else set()
    answer_types: set[str] = set()
    for answer in answers:
        if answer is None:
            answer_types.add("missing")
        else:
            answer_types.add("str")

    return SplitAudit(
        split=split,
        path=path.as_posix(),
        status="present",
        error=None,
        record_count=len(records),
        unique_id_count=len(unique_ids),
        duplicate_id_count=max(stats.duplicate_id_count, len(ids) - len(unique_ids)),
        blank_question_count=blank_questions,
        blank_answer_count=blank_answers,
        missing_answer_count=missing_answers,
        question_type=",".join(sorted(question_types)) or "absent",
        answer_type=",".join(sorted(answer_types)) or "absent",
        question_length_chars=length_stats([len(text) for text in questions]),
        answer_length_chars=length_stats(answer_lengths_chars),
        question_length_tokens=length_stats(
            [len(tokenize_legal_text(text)) for text in questions]
        ),
        answer_length_tokens=length_stats(answer_lengths_tokens),
        exact_duplicate_questions=exact_dupes,
        normalized_duplicate_questions=normalized_dupes,
        duplicate_qa_pairs=duplicate_qa,
        ids=tuple(sorted(unique_ids)),
        questions_by_id={record.id: record.question for record in records},
        answers_by_id={record.id: record.answer for record in records},
    )


def cross_split_overlaps(splits: Mapping[str, SplitAudit]) -> dict[str, Any]:
    """Report ID and normalized-question overlaps across present splits."""

    present = {
        name: audit
        for name, audit in splits.items()
        if audit.status == "present" and audit.record_count > 0
    }
    id_overlaps: dict[str, list[str]] = {}
    question_overlaps: dict[str, list[str]] = {}
    names = sorted(present)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = present[left_name]
            right = present[right_name]
            key = f"{left_name}__{right_name}"
            shared_ids = sorted(set(left.ids) & set(right.ids))
            id_overlaps[key] = shared_ids
            left_norm = {
                normalize_question_text(text): question_id
                for question_id, text in left.questions_by_id.items()
            }
            right_norm = {
                normalize_question_text(text): question_id
                for question_id, text in right.questions_by_id.items()
            }
            shared_questions = sorted(set(left_norm) & set(right_norm))
            question_overlaps[key] = shared_questions

    forbidden_pair_sets = {
        frozenset({"train", "warmup"}),
        frozenset({"train", "public"}),
        frozenset({"train", "private"}),
        frozenset({"warmup", "public"}),
        frozenset({"warmup", "private"}),
        frozenset({"public", "private"}),
    }

    def _is_forbidden(pair_key: str) -> bool:
        left, right = pair_key.split("__", 1)
        return frozenset({left, right}) in forbidden_pair_sets

    forbidden_id_hits = {
        pair: ids for pair, ids in id_overlaps.items() if _is_forbidden(pair) and ids
    }
    forbidden_question_hits = {
        pair: values
        for pair, values in question_overlaps.items()
        if _is_forbidden(pair) and values
    }
    return {
        "present_splits": names,
        "id_overlaps": {key: list(values) for key, values in id_overlaps.items()},
        "id_overlap_counts": {key: len(values) for key, values in id_overlaps.items()},
        "normalized_question_overlaps": {
            key: list(values) for key, values in question_overlaps.items()
        },
        "normalized_question_overlap_counts": {
            key: len(values) for key, values in question_overlaps.items()
        },
        "forbidden_id_overlap_counts": {
            key: len(values) for key, values in forbidden_id_hits.items()
        },
        "forbidden_normalized_question_overlap_counts": {
            key: len(values) for key, values in forbidden_question_hits.items()
        },
        "forbidden_overlap_total": sum(len(v) for v in forbidden_id_hits.values())
        + sum(len(v) for v in forbidden_question_hits.values()),
    }


def training_overlap_remediation(
    splits: Mapping[str, SplitAudit],
    *,
    overlap_policy: Literal["fail", "exclude_and_record"],
    remediation_id: str | None,
) -> tuple[dict[str, Any], tuple[TrainingOverlapExclusion, ...]]:
    """Assess the effective train set without rewriting any source split."""

    if overlap_policy == "exclude_and_record" and not remediation_id:
        raise DataFeasibilityAuditError(
            "exclude_and_record requires a non-blank remediation_id"
        )
    train = splits.get("train")
    train_questions = (
        train.questions_by_id if train is not None and train.status == "present" else {}
    )
    comparisons = {
        split: audit.questions_by_id
        for split, audit in splits.items()
        if split != "train" and audit.status == "present"
    }
    exclusions = derive_train_overlap_exclusions(train_questions, comparisons)
    reason_counts = Counter(
        reason for exclusion in exclusions for reason in exclusion.reasons
    )
    source_train_count = len(train_questions)
    excluded_count = len(exclusions)
    effective_train_count = (
        source_train_count - excluded_count
        if overlap_policy == "exclude_and_record"
        else source_train_count
    )
    remaining_overlap_cases = (
        0 if overlap_policy == "exclude_and_record" else excluded_count
    )
    return (
        {
            "policy": overlap_policy,
            "remediation_id": remediation_id,
            "source_train_count": source_train_count,
            "source_train_overlap_case_count": excluded_count,
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "effective_train_count": effective_train_count,
            "effective_train_to_nontrain_overlap_case_count": remaining_overlap_cases,
            "status": (
                "remediated"
                if overlap_policy == "exclude_and_record" and excluded_count
                else "clean"
                if not excluded_count
                else "blocked"
            ),
        },
        exclusions,
    )


def _contains_answer(haystack: str, answer: str) -> bool:
    if not answer.strip():
        return False
    return answer in haystack


def _normalized_contains(haystack: str, needle: str) -> bool:
    if not needle.strip():
        return False
    return normalize_question_text(needle) in normalize_question_text(haystack)


def _token_overlap(answer: str, evidence: str) -> float:
    answer_tokens = set(tokenize_legal_text(answer))
    if not answer_tokens:
        return 0.0
    evidence_tokens = set(tokenize_legal_text(evidence))
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _estimate_prompt_tokens(
    question: str, evidence: PackedEvidence, builder: PromptBuilder
) -> int:
    rendered = builder.build_rag(question, evidence).text
    return len(tokenize_legal_text(rendered))


def audit_index_metadata_for_gold(
    index: BM25Index,
    *,
    gold_answers: Sequence[str] = (),
) -> dict[str, Any]:
    """Ensure indexed documents expose no answer-shaped fields or gold text."""

    field_names = set(BM25Index.__dataclass_fields__) | set(
        type(index.documents[0]).__dataclass_fields__ if index.documents else ()
    )
    forbidden_fields = sorted(field_names & _FORBIDDEN_INDEX_KEYS)
    gold_hits: list[str] = []
    for document in index.documents:
        payload = json.dumps(asdict(document), ensure_ascii=False)
        for answer in gold_answers:
            if answer.strip() and answer in payload:
                gold_hits.append(document.chunk_id)
                break
            if answer.strip() and answer in document.retrieval_text:
                gold_hits.append(document.chunk_id)
                break
    return {
        "forbidden_field_names": forbidden_fields,
        "forbidden_field_count": len(forbidden_fields),
        "gold_text_in_index_document_count": len(gold_hits),
        "gold_text_in_index_chunk_ids": sorted(set(gold_hits)),
        "pass": len(forbidden_fields) == 0 and len(gold_hits) == 0,
    }


def audit_training_path_policy(
    candidate_splits: Sequence[SplitName],
) -> dict[str, Any]:
    """Training builders may only consume the train split."""

    disallowed = sorted(
        {split for split in candidate_splits if split not in TRAINING_ALLOWED_SPLITS}
    )
    return {
        "allowed_splits": sorted(TRAINING_ALLOWED_SPLITS),
        "candidate_splits": list(candidate_splits),
        "disallowed_splits": disallowed,
        "pass": not disallowed,
    }


def audit_reference_access_policy() -> dict[str, Any]:
    """Public/private reference answers must stay unavailable to training."""

    public_blocked = False
    private_blocked = False
    public_error: str | None = None
    private_error: str | None = None
    try:
        load_questions(Path("unused"), split="public", include_answers=True)
    except QuestionLoadError as exc:
        public_blocked = "not permitted" in str(exc).lower()
        public_error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        public_error = str(exc)

    try:
        load_questions(Path("unused"), split="private", include_answers=True)
    except QuestionLoadError as exc:
        private_blocked = "not permitted" in str(exc).lower()
        private_error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        private_error = str(exc)

    return {
        "public_answer_load_blocked": public_blocked,
        "private_answer_load_blocked": private_blocked,
        "public_error": public_error,
        "private_error": private_error,
        "warmup_reference_access": SPLIT_USAGE_REGISTRY["warmup"].reference_access,
        "train_reference_access": SPLIT_USAGE_REGISTRY["train"].reference_access,
        "pass": public_blocked and private_blocked,
    }


def _retrieve_case(
    question: LegalQuestion,
    *,
    index: BM25Index,
    chunks: Mapping[str, LegalChunk],
    documents: Mapping[str, LegalDocument] | None,
    config: ProjectConfig,
    prompt_builder: PromptBuilder,
    max_seq_tokens: int,
) -> CaseRetrievalAudit:
    query = question.question
    answer = question.answer or ""
    gold_in_query = bool(answer.strip()) and answer in query
    # Query identity for retrieval is question-only by construction.
    # Uses frozen rough_top_n / evidence budget; BM25 order only (no semantic
    # reranker), matching B2's explicit optional-reranker fallback path.
    hits = retrieve_bm25(index, query, top_k=config.retrieval.rough_top_n)
    packed: PackedEvidence | None = None
    zero_evidence = len(hits) == 0
    try:
        if hits:
            deduplicated = deduplicate_retrieved_chunks(hits, chunks)
            selected_hits = deduplicated.kept_hits[: config.evidence.evidence_top_k]
            packed = pack_evidence(
                selected_hits,
                chunks,
                max_total_chars=config.evidence.max_total_chars,
                max_chunks_per_document=config.evidence.max_chunks_per_document,
                documents=documents,
            )
            zero_evidence = len(packed.included_ids) == 0
    except Exception:
        packed = None
        zero_evidence = True

    evidence_text = packed.rendered_text if packed is not None else ""
    top1_text = ""
    top3_text = ""
    topk_text = evidence_text
    if packed is not None and packed.included_hits:
        included = packed.included_hits
        top1_ids = {included[0].chunk_id}
        top3_ids = {hit.chunk_id for hit in included[:3]}
        top1_text = "\n".join(
            chunks[hit.chunk_id].raw_text
            for hit in included
            if hit.chunk_id in top1_ids and hit.chunk_id in chunks
        )
        top3_text = "\n".join(
            chunks[hit.chunk_id].raw_text
            for hit in included
            if hit.chunk_id in top3_ids and hit.chunk_id in chunks
        )

    overlap = (
        _token_overlap(answer, evidence_text) if answer and evidence_text else None
    )
    low_overlap = overlap is not None and overlap < LOW_OVERLAP_THRESHOLD
    evidence_tokens = len(tokenize_legal_text(evidence_text)) if evidence_text else 0
    target_tokens = len(tokenize_legal_text(answer)) if answer else 0
    if packed is not None and answer:
        prompt_tokens = _estimate_prompt_tokens(query, packed, prompt_builder)
        combined = prompt_tokens + target_tokens
        # Fit policy: question+target must fit; evidence already packed under budget.
        scaffolding_and_target = len(tokenize_legal_text(query)) + target_tokens + 32
        if scaffolding_and_target > max_seq_tokens:
            fit_status = "TARGET_DOES_NOT_FIT"
            fits = False
        else:
            fits = combined <= max_seq_tokens
            fit_status = "fits" if fits else "evidence_truncated_needed"
    elif not answer:
        prompt_tokens = None
        combined = None
        fits = None
        fit_status = "no_target"
    else:
        prompt_tokens = None
        combined = None
        fits = None
        fit_status = "no_evidence"

    doc_ids = tuple(
        sorted(
            {
                chunks[hit.chunk_id].document_id
                for hit in (packed.included_hits if packed else ())
                if hit.chunk_id in chunks
            }
        )
    )
    gold_in_evidence = bool(answer.strip()) and answer in evidence_text

    split_label: SplitName = (
        question.split if question.split in DEFAULT_SPLIT_PATHS else "train"
    )
    return CaseRetrievalAudit(
        id=question.id,
        split=split_label,
        query=query,
        status="ok" if packed is not None else "no_evidence",
        top1_exact_substring=_contains_answer(top1_text, answer) if answer else None,
        top3_exact_substring=_contains_answer(top3_text, answer) if answer else None,
        topk_exact_substring=_contains_answer(topk_text, answer) if answer else None,
        top1_normalized_substring=(
            _normalized_contains(top1_text, answer) if answer else None
        ),
        top3_normalized_substring=(
            _normalized_contains(top3_text, answer) if answer else None
        ),
        topk_normalized_substring=(
            _normalized_contains(topk_text, answer) if answer else None
        ),
        answer_evidence_token_overlap=overlap,
        zero_evidence=zero_evidence,
        low_overlap=low_overlap,
        retrieved_document_ids=doc_ids,
        evidence_token_count=evidence_tokens if packed is not None else None,
        prompt_token_count=prompt_tokens,
        target_token_count=target_tokens if answer else None,
        prompt_plus_target_tokens=combined,
        fits_provisional_max_seq=fits,
        fit_status=fit_status,
        gold_in_query=gold_in_query,
        gold_in_evidence=gold_in_evidence,
    )


def run_retrieval_support_audit(
    questions: Sequence[LegalQuestion],
    *,
    index: BM25Index,
    chunks: Mapping[str, LegalChunk],
    documents: Mapping[str, LegalDocument] | None,
    config: ProjectConfig,
    prompt_builder: PromptBuilder,
    max_seq_tokens: int = PROVISIONAL_MAX_SEQ_TOKENS,
) -> tuple[dict[str, Any], tuple[CaseRetrievalAudit, ...]]:
    """Run frozen-B2 retrieval diagnostics for answered questions."""

    ordered = tuple(sorted(questions, key=lambda item: item.id))
    cases = tuple(
        _retrieve_case(
            question,
            index=index,
            chunks=chunks,
            documents=documents,
            config=config,
            prompt_builder=prompt_builder,
            max_seq_tokens=max_seq_tokens,
        )
        for question in ordered
        if question.answer is not None
    )
    if not cases:
        return (
            {
                "status": "skipped",
                "reason": "no answered questions for retrieval diagnostics",
                "case_count": 0,
            },
            (),
        )

    def _rate(predicate: Any) -> float:
        return sum(1 for case in cases if predicate(case)) / len(cases)

    evidence_token_values = [
        case.evidence_token_count
        for case in cases
        if case.evidence_token_count is not None
    ]
    prompt_token_values = [
        case.prompt_token_count for case in cases if case.prompt_token_count is not None
    ]
    fit_known = [case for case in cases if case.fits_provisional_max_seq is not None]
    summary = {
        "status": "ok",
        "case_count": len(cases),
        "zero_evidence_rate": _rate(lambda case: case.zero_evidence),
        "low_overlap_rate": _rate(lambda case: bool(case.low_overlap)),
        "top1_exact_substring_rate": _rate(
            lambda case: bool(case.top1_exact_substring)
        ),
        "top3_exact_substring_rate": _rate(
            lambda case: bool(case.top3_exact_substring)
        ),
        "topk_exact_substring_rate": _rate(
            lambda case: bool(case.topk_exact_substring)
        ),
        "top1_normalized_substring_rate": _rate(
            lambda case: bool(case.top1_normalized_substring)
        ),
        "top3_normalized_substring_rate": _rate(
            lambda case: bool(case.top3_normalized_substring)
        ),
        "topk_normalized_substring_rate": _rate(
            lambda case: bool(case.topk_normalized_substring)
        ),
        "mean_answer_evidence_token_overlap": (
            sum(case.answer_evidence_token_overlap or 0.0 for case in cases)
            / len(cases)
        ),
        "retrieved_document_diversity_mean": (
            sum(len(case.retrieved_document_ids) for case in cases) / len(cases)
        ),
        "evidence_token_length": length_stats(
            [value for value in evidence_token_values if value is not None]
        ).as_dict(),
        "prompt_token_length": length_stats(
            [value for value in prompt_token_values if value is not None]
        ).as_dict(),
        "provisional_max_seq_tokens": max_seq_tokens,
        "provisional_max_seq_note": (
            "Local tokenize_legal_text proxy only; official tokenizer pending FTR-04"
        ),
        "prompt_plus_target_fit_rate": (
            sum(1 for case in fit_known if case.fits_provisional_max_seq)
            / len(fit_known)
            if fit_known
            else None
        ),
        "target_does_not_fit_rate": (
            sum(1 for case in cases if case.fit_status == "TARGET_DOES_NOT_FIT")
            / len(cases)
        ),
        "gold_in_query_count": sum(1 for case in cases if case.gold_in_query),
        "gold_in_evidence_count": sum(1 for case in cases if case.gold_in_evidence),
    }
    return summary, cases


def _source_integrity(repo_root: Path) -> dict[str, Any]:
    from scripts.verify_data_manifest import (
        DEFAULT_MANIFEST_PATH,
        build_manifest,
        verify_manifest,
    )

    root = repo_root.resolve()
    try:
        manifest = build_manifest(root)
        verify_manifest(root, root / DEFAULT_MANIFEST_PATH)
        digest = hashlib.sha256(
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return {
            "status": "verified",
            "data_manifest_hash": digest,
            "file_count": len(manifest.get("files", [])),
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_data_feasibility_audit(
    repo_root: str | Path,
    *,
    split_paths: Mapping[SplitName, Path] | None = None,
    index: BM25Index | None = None,
    chunks: Mapping[str, LegalChunk] | None = None,
    documents: Mapping[str, LegalDocument] | None = None,
    config: ProjectConfig | None = None,
    prompt_builder: PromptBuilder | None = None,
    training_candidate_splits: Sequence[SplitName] = ("train",),
    overlap_policy: Literal["fail", "exclude_and_record"] = "fail",
    overlap_remediation_id: str | None = None,
    max_seq_tokens: int = PROVISIONAL_MAX_SEQ_TOKENS,
    require_complete_freeze: bool = False,
) -> AuditResult:
    """Run the read-only FTR-03 audit. Never writes source data or training sets."""

    root = Path(repo_root).resolve()
    paths = {
        split: root / relative
        for split, relative in (split_paths or DEFAULT_SPLIT_PATHS).items()
    }
    freeze = load_b2_freeze_fingerprint(root)
    freeze_info = {
        "status": freeze.status,
        "config_hash": freeze.config_hash,
        "index_fingerprint": freeze.index_fingerprint,
        "prompt_hash": freeze.prompt_hash,
        "frozen_config_path": freeze.frozen_config_path,
    }
    if require_complete_freeze:
        require_complete_b2_freeze(freeze)

    selected_config = config or load_config(root / DEFAULT_FROZEN_CONFIG_REL)
    selected_builder = prompt_builder or PromptBuilder.from_config(
        selected_config.prompts,
        prompt_dir=root / "configs" / "prompts",
    )

    split_audits: dict[str, SplitAudit] = {}
    for split, path in sorted(paths.items()):
        include_answers = split in {"train", "warmup"}
        split_audits[split] = audit_split_file(
            path, split=split, include_answers=include_answers
        )

    overlaps = cross_split_overlaps(split_audits)
    training_remediation, training_exclusions = training_overlap_remediation(
        split_audits,
        overlap_policy=overlap_policy,
        remediation_id=overlap_remediation_id,
    )
    nontraining_overlap_total = sum(
        value
        for pair, value in overlaps["forbidden_id_overlap_counts"].items()
        if "train" not in pair.split("__")
    ) + sum(
        value
        for pair, value in overlaps[
            "forbidden_normalized_question_overlap_counts"
        ].items()
        if "train" not in pair.split("__")
    )
    training_remediation["unremediated_nontraining_overlap_total"] = (
        nontraining_overlap_total
    )
    integrity = _source_integrity(root)

    retrieval_summary: dict[str, Any]
    per_case: tuple[CaseRetrievalAudit, ...] = ()
    index_leakage: dict[str, Any]
    if index is not None and chunks is not None:
        train_records = []
        if split_audits["train"].status == "present":
            train_records = list(
                load_questions(paths["train"], split="train", include_answers=True)
            )
        # Prefer train for grounding diagnostics; else answered warmup for smoke only.
        questions_for_retrieval: list[LegalQuestion] = train_records
        if not questions_for_retrieval and split_audits["warmup"].status == "present":
            questions_for_retrieval = list(
                load_questions(paths["warmup"], split="warmup", include_answers=True)
            )
        golds = [q.answer for q in questions_for_retrieval if q.answer]
        index_leakage = audit_index_metadata_for_gold(
            index, gold_answers=[g for g in golds if g]
        )
        retrieval_summary, per_case = run_retrieval_support_audit(
            questions_for_retrieval,
            index=index,
            chunks=chunks,
            documents=documents,
            config=selected_config,
            prompt_builder=selected_builder,
            max_seq_tokens=max_seq_tokens,
        )
        retrieval_summary["index_fingerprint"] = index.index_fingerprint
        retrieval_summary["bm25_config"] = index.config.as_dict()
    else:
        index_leakage = {
            "pass": None,
            "status": "skipped",
            "reason": (
                "selected-contexts / BM25 index unavailable for competition audit"
            ),
        }
        retrieval_summary = {
            "status": "blocked",
            "reason": (
                "Frozen B2 retrieval diagnostics require selected-contexts.zip "
                "and a built BM25 index; corpus currently missing"
            ),
            "zero_evidence_rate": None,
            "prompt_plus_target_fit_rate": None,
            "provisional_max_seq_tokens": max_seq_tokens,
        }

    query_leakage_count = sum(1 for case in per_case if case.gold_in_query)
    reference_policy = audit_reference_access_policy()
    training_policy = audit_training_path_policy(training_candidate_splits)

    leakage = {
        "gold_in_retrieval_query": {
            "count": query_leakage_count,
            "pass": query_leakage_count == 0 if per_case else None,
            "note": (
                "Query is question-only by construction; count flags answers that "
                "are substrings of the question text itself"
            ),
        },
        "gold_in_index_or_chunk_metadata": index_leakage,
        "public_private_reference_access": reference_policy,
        "training_paths_train_only": training_policy,
        "pass": all(
            value is True
            for value in (
                reference_policy["pass"],
                training_policy["pass"],
                index_leakage.get("pass") in {True, None},
                (query_leakage_count == 0 if per_case else True),
            )
        ),
    }

    hard_stop_reasons: list[str] = []
    train = split_audits["train"]
    if train.status != "present":
        hard_stop_reasons.append("train split file missing; no train answers available")
    elif train.record_count == 0:
        hard_stop_reasons.append("train split is empty")
    elif train.missing_answer_count == train.record_count:
        hard_stop_reasons.append("train split has no answers")
    elif train.blank_answer_count > 0 and train.blank_answer_count == (
        train.record_count - train.missing_answer_count
    ):
        hard_stop_reasons.append("all available train answers are blank")

    if training_remediation["effective_train_to_nontrain_overlap_case_count"] > 0:
        hard_stop_reasons.append("effective train split still has cross-split overlaps")
    if training_remediation["effective_train_count"] <= 0:
        hard_stop_reasons.append(
            "effective train split is empty after overlap remediation"
        )
    if integrity.get("status") != "verified":
        hard_stop_reasons.append("source data integrity verification failed")
    if freeze.index_fingerprint == UNRESOLVED or freeze.status != "complete":
        hard_stop_reasons.append(
            "frozen B2 corpus/index fingerprints unresolved; "
            "retrieval control incomplete"
        )
    if leakage["pass"] is False:
        hard_stop_reasons.append("leakage policy checks failed")

    fit_section = {
        "provisional_max_seq_tokens": max_seq_tokens,
        "estimator": "tokenize_legal_text_local_proxy",
        "prompt_plus_target_fit_rate": retrieval_summary.get(
            "prompt_plus_target_fit_rate"
        ),
        "target_does_not_fit_rate": retrieval_summary.get("target_does_not_fit_rate"),
        "status": retrieval_summary.get("status"),
    }

    hard_stop = bool(hard_stop_reasons)
    status = "hard_stop" if hard_stop else "pass"
    return AuditResult(
        schema_version=AUDIT_SCHEMA_VERSION,
        status=status,
        hard_stop=hard_stop,
        hard_stop_reasons=tuple(hard_stop_reasons),
        b2_freeze=freeze_info,
        splits={name: audit.public_dict() for name, audit in split_audits.items()},
        cross_split=overlaps,
        training_remediation=training_remediation,
        training_exclusions=training_exclusions,
        retrieval=retrieval_summary,
        fit=fit_section,
        leakage=leakage,
        source_integrity=integrity,
        per_case=per_case,
    )


def write_audit_artifacts(
    result: AuditResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic audit artifacts without touching source data."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    per_case_path = directory / "per_case.jsonl"
    leakage_path = directory / "leakage_report.json"
    exclusions_path = directory / "training_exclusions.jsonl"

    summary_path.write_text(
        json.dumps(result.summary_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        json.dumps(case.as_dict(), ensure_ascii=False, sort_keys=True)
        for case in sorted(result.per_case, key=lambda item: (item.split, item.id))
    ]
    per_case_path.write_text(
        ("\n".join(lines) + ("\n" if lines else "")),
        encoding="utf-8",
    )
    leakage_path.write_text(
        json.dumps(result.leakage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exclusions_path.write_text(
        "".join(
            json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in result.training_exclusions
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "per_case": per_case_path,
        "leakage_report": leakage_path,
        "training_exclusions": exclusions_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for the repository audit."""

    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Read-only FTR-03 data feasibility and leakage audit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/finetuned_reader_train.yaml"),
        help="Generative training profile defining the overlap remediation policy.",
    )
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="Explicitly run the expensive frozen-B2 retrieval diagnostics.",
    )
    args = parser.parse_args(argv)
    try:
        training_config = load_config(root / args.config)
        settings = training_config.finetuned_reader
        if settings is None or training_config.data.split != "train":
            raise DataFeasibilityAuditError(
                "FTR-03 requires a generative training config with data.split='train'"
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    retrieval_kwargs: dict[str, Any] = {}
    retrieval_error: str | None = None
    try:
        # The CLI is the explicit, potentially expensive path. Unit callers keep
        # the injectable read-only API and may provide a fixture index instead.
        if args.with_retrieval:
            from ..pipeline import prepare_bm25_index_from_config

            preparation = prepare_bm25_index_from_config(
                root / DEFAULT_FROZEN_CONFIG_REL,
                repo_root=root,
                rebuild_index=False,
            )
            retrieval_kwargs = {
                "index": preparation.index,
                "chunks": {chunk.chunk_id: chunk for chunk in preparation.chunks},
                "documents": preparation.documents,
                "config": preparation.config,
                "prompt_builder": PromptBuilder.from_config(
                    preparation.config.prompts,
                    prompt_dir=root / "configs" / "prompts",
                ),
            }
    except Exception as exc:
        # The audit remains useful for schema/leakage checks when B2 assets are
        # unavailable; the blocked status is retained and never hidden.
        retrieval_kwargs = {}
        retrieval_error = f"{type(exc).__name__}: {exc}"
    result = run_data_feasibility_audit(
        root,
        overlap_policy=settings.overlap_policy,
        overlap_remediation_id=settings.overlap_remediation_id,
        **retrieval_kwargs,
    )
    if not args.with_retrieval:
        result = replace(
            result,
            retrieval={
                "status": "not_run",
                "reason": "Use --with-retrieval for the expensive frozen-B2 audit",
            },
            fit={
                "status": "not_run",
                "reason": "Tokenizer/evidence fit is assessed in FTR-05/FTR-06",
            },
        )
    if retrieval_error is not None:
        result = replace(
            result,
            retrieval={**result.retrieval, "error": retrieval_error},
        )
    paths = write_audit_artifacts(result, root / DEFAULT_AUDIT_DIR)
    print(
        json.dumps(
            {
                "status": result.status,
                "hard_stop": result.hard_stop,
                "hard_stop_reasons": list(result.hard_stop_reasons),
                "artifacts": {name: path.as_posix() for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result.hard_stop else 0


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditResult",
    "CaseRetrievalAudit",
    "DEFAULT_AUDIT_DIR",
    "DEFAULT_SPLIT_PATHS",
    "DataFeasibilityAuditError",
    "PROVISIONAL_MAX_SEQ_TOKENS",
    "SplitAudit",
    "TRAINING_ALLOWED_SPLITS",
    "audit_index_metadata_for_gold",
    "audit_reference_access_policy",
    "audit_split_file",
    "audit_training_path_policy",
    "cross_split_overlaps",
    "length_stats",
    "main",
    "normalize_question_text",
    "run_data_feasibility_audit",
    "run_retrieval_support_audit",
    "training_overlap_remediation",
    "write_audit_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
