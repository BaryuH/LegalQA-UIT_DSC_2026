"""Leakage-aware synthetic Vietnamese legal-query generation (TASK 09).

The default generator is deterministic and dependency-light so the local
scaffold can exercise the schema and filters without model downloads.  The
Transformers backend is explicit and requires live CUDA; it never silently
falls back to the template backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol

from legal_rag.schemas import CanonicalID, DomainModel, NonBlankText
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.cuda_policy import require_live_cuda
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.query.citation_parser import parse_citations

QueryType = Literal[
    "direct",
    "citizen_paraphrase",
    "scenario",
    "citation_free",
    "condition_exception",
]
DocumentSplit = Literal["train", "validation", "test"]

QUERY_TYPES: tuple[QueryType, ...] = (
    "direct",
    "citizen_paraphrase",
    "scenario",
    "citation_free",
    "condition_exception",
)
DEFAULT_PROMPT_VERSION = "sedar-task09-v1"
SYNTHETIC_SCHEMA_VERSION = "sedar-retrieval-v3-synthetic-query-v1"
GENERATOR_PROMPT = """\
You generate exactly one Vietnamese legal question supported by the supplied
passage. Do not answer it. Do not invent citations, dates, legal documents, or
facts. Return only the question and preserve the requested query type.

Query type: {query_type}
Passage:
{passage}
"""
_VIETNAMESE_MARKERS = set("ăâđêôơưĂÂĐÊÔƠƯ")
_TRIVIAL_WORDS = {
    "ai",
    "cái",
    "gi",
    "gì",
    "nao",
    "nào",
    "quy",
    "định",
}
_STOPWORDS = {
    "các",
    "cho",
    "của",
    "được",
    "là",
    "nào",
    "người",
    "những",
    "pháp",
    "theo",
    "trong",
    "và",
    "về",
}


class SyntheticQueryError(ValueError):
    """Raised when synthetic-query generation violates its contract."""


class QueryGenerator(Protocol):
    """Backend contract for one passage-conditioned query."""

    model: str
    revision: str

    def generate(
        self, passage: CanonicalPassage, query_type: QueryType
    ) -> GeneratedQuery: ...


class SyntheticQueryRecord(DomainModel):
    """One accepted training-only synthetic query record."""

    synthetic_id: CanonicalID
    query: NonBlankText
    positive_passage_id: CanonicalID
    source_document_id: CanonicalID
    query_type: QueryType
    legal_domain: NonBlankText
    generator_model: NonBlankText
    generator_revision: NonBlankText
    prompt_version: NonBlankText
    source_hash: NonBlankText
    quality_flags: tuple[NonBlankText, ...]
    raw_generation: NonBlankText
    source_split: DocumentSplit = "train"


class SyntheticRejectionRecord(DomainModel):
    """Metadata for a rejected attempt without persisting generated text."""

    passage_id: CanonicalID
    source_document_id: CanonicalID
    query_type: QueryType
    source_hash: NonBlankText
    rejection_reason: NonBlankText


class SyntheticGenerationErrorRecord(DomainModel):
    """Provider failure metadata without prompt or model-output text."""

    passage_id: CanonicalID
    source_document_id: CanonicalID
    query_type: QueryType
    error_type: NonBlankText
    error_message: NonBlankText


@dataclass(frozen=True, slots=True)
class GeneratedQuery:
    """Backend output before quality and leakage filters."""

    query: str
    raw_generation: str


@dataclass(frozen=True, slots=True)
class SyntheticGenerationConfig:
    """Deterministic generation and split policy."""

    target_accepted: int = 10_000
    max_attempts: int = 30_000
    query_types: tuple[QueryType, ...] = QUERY_TYPES
    seed: int = 42
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    source_split: DocumentSplit = "train"
    legal_domain: str = "vietnamese_law"
    prompt_version: str = DEFAULT_PROMPT_VERSION
    min_source_token_overlap: float = 0.15

    def __post_init__(self) -> None:
        if self.target_accepted <= 0:
            raise ValueError("target_accepted must be positive")
        if self.max_attempts < self.target_accepted:
            raise ValueError("max_attempts must be >= target_accepted")
        if not self.query_types:
            raise ValueError("query_types must not be empty")
        if (
            self.validation_fraction < 0
            or self.test_fraction < 0
            or self.validation_fraction + self.test_fraction >= 1
        ):
            raise ValueError("validation/test fractions must sum to less than one")
        if not 0 <= self.min_source_token_overlap <= 1:
            raise ValueError("min_source_token_overlap must be between zero and one")


@dataclass(frozen=True, slots=True)
class SyntheticBuildReport:
    """Content-free generation and filter counters for the pilot audit."""

    schema_version: str
    attempts: int
    accepted: int
    target_accepted: int
    source_document_count: int
    train_document_count: int
    validation_document_count: int
    test_document_count: int
    source_split_leakage_count: int
    generation_error_count: int
    source_split: DocumentSplit
    document_split_source: Literal["deterministic_hash", "external_manifest"]
    rejection_counts: dict[str, int]
    document_isolation_ok: bool
    prompt_version: str
    prompt_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "target_accepted": self.target_accepted,
            "acceptance_rate": (
                self.accepted / self.attempts if self.attempts else 0.0
            ),
            "source_document_count": self.source_document_count,
            "train_document_count": self.train_document_count,
            "validation_document_count": self.validation_document_count,
            "test_document_count": self.test_document_count,
            "source_split_leakage_count": self.source_split_leakage_count,
            "generation_error_count": self.generation_error_count,
            "source_split": self.source_split,
            "document_split_source": self.document_split_source,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "document_isolation_ok": self.document_isolation_ok,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "heuristic_gate_metrics": {
                "supportable_lower_bound": (
                    self.accepted / self.attempts if self.attempts else 0.0
                ),
                "valid_vietnamese_lower_bound": (
                    self.accepted / self.attempts if self.attempts else 0.0
                ),
                "wrong_citation_observed_rate": (
                    self.rejection_counts.get("wrong_citation", 0) / self.attempts
                    if self.attempts
                    else 0.0
                ),
                "interpretation": (
                    "Machine-filter diagnostics only; manual audit gate remains "
                    "mandatory."
                ),
            },
            "manual_audit_required": True,
        }


def synthetic_prompt_sha256() -> str:
    """Return the hash of the frozen generator prompt template."""

    return sha256(GENERATOR_PROMPT.encode("utf-8")).hexdigest()


def assign_document_splits(
    document_ids: Iterable[str],
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, DocumentSplit]:
    """Assign whole documents to disjoint deterministic train/val/test groups."""

    if (
        validation_fraction < 0
        or test_fraction < 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("validation/test fractions must sum to less than one")
    unique_ids = sorted(set(str(item) for item in document_ids))
    assignments: dict[str, DocumentSplit] = {}
    for document_id in unique_ids:
        digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64)
        if bucket < test_fraction:
            assignments[document_id] = "test"
        elif bucket < test_fraction + validation_fraction:
            assignments[document_id] = "validation"
        else:
            assignments[document_id] = "train"
    return assignments


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _sanitize_generated_query(text: str) -> str:
    """Keep one question line and discard hidden/reasoning-style sections."""

    without_thinking = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    lines = [
        re.sub(
            r"^(?:câu hỏi|question)\s*:\s*",
            "",
            line.strip().strip("`*_-'\""),
            flags=re.IGNORECASE,
        ).strip()
        for line in without_thinking.splitlines()
        if line.strip()
    ]
    if not lines:
        return ""
    question_line = next((line for line in lines if "?" in line), lines[0])
    if "?" in question_line:
        question_line = question_line[: question_line.index("?") + 1]
    return " ".join(question_line.split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", _normalize_text(text), flags=re.UNICODE))


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS}


def _topic_from_passage(passage: CanonicalPassage) -> str:
    text = re.sub(r"\[[A-Z _]+\]", " ", passage.reader_text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"^điều\s+\d+[A-Za-z]?(?:\s*[\.:;-])?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    words = sentence.split()[:18]
    topic = " ".join(words).strip(" .,;:")
    if topic:
        return topic
    if passage.article_title:
        return passage.article_title
    return "nội dung của quy định này"


class TemplateQueryGenerator:
    """Deterministic local scaffold generator with explicit provenance."""

    model = "deterministic-template"
    revision = "template-v1"

    def generate(
        self, passage: CanonicalPassage, query_type: QueryType
    ) -> GeneratedQuery:
        topic = _topic_from_passage(passage)
        article = (
            f"Điều {passage.article_number}"
            if passage.article_number
            else "quy định này"
        )
        templates = {
            "direct": f"{article} quy định chính sách gì về {topic}?",
            "citizen_paraphrase": (
                f"Người dân cần biết gì khi áp dụng quy định về {topic}?"
            ),
            "scenario": (
                f"Nếu phát sinh trường hợp liên quan đến {topic}, "
                "cần thực hiện như thế nào?"
            ),
            "citation_free": f"Quy định pháp luật nào điều chỉnh việc {topic}?",
            "condition_exception": (
                f"Điều kiện, giới hạn hoặc ngoại lệ nào áp dụng khi {topic}?"
            ),
        }
        query = templates[query_type]
        return GeneratedQuery(query=query, raw_generation=query)


class TransformersQueryGenerator:
    """Explicit CUDA Transformers backend for server-side query generation."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        device: str = "cuda",
        dtype: str = "bf16",
        max_input_tokens: int = 4096,
        max_new_tokens: int = 96,
        local_files_only: bool = False,
    ) -> None:
        if not revision.strip():
            raise ValueError("revision must be pinned")
        if not device.startswith("cuda"):
            raise SyntheticQueryError(
                "The Transformers generator requires an explicit CUDA device."
            )
        require_live_cuda("TASK 09 synthetic query generation")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SyntheticQueryError(
                "Transformers TASK 09 backend requires torch and transformers."
            ) from exc
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be one of: bf16, fp16, fp32")
        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        self.model_name = model
        self.model = model
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_model: Any = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self.language_model.eval()
        self.input_device = next(self.language_model.parameters()).device

    def generate(
        self, passage: CanonicalPassage, query_type: QueryType
    ) -> GeneratedQuery:
        prompt = GENERATOR_PROMPT.format(
            query_type=query_type,
            passage=passage.reader_text[:12_000],
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        output = self.language_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        raw_value = self.tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        )
        raw = str(raw_value).strip()
        query = _sanitize_generated_query(raw)
        return GeneratedQuery(query=query, raw_generation=query)


class _QueryDeduplicator:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._buckets: defaultdict[tuple[int, str], list[str]] = defaultdict(list)

    @staticmethod
    def _comparison_text(query: str) -> str:
        normalized = _normalize_text(query)
        return re.sub(r"\b\d+\b", "<num>", normalized)

    def _bucket(self, query: str) -> tuple[int, str]:
        comparison = self._comparison_text(query)
        tokens = _tokens(comparison)
        return (len(comparison) // 20, " ".join(tokens[:3]))

    def reason(self, query: str) -> str | None:
        normalized = _normalize_text(query)
        if normalized in self._seen:
            return "duplicate"
        comparison = self._comparison_text(query)
        bucket = self._bucket(query)
        for candidate in self._buckets[bucket]:
            if SequenceMatcher(None, comparison, candidate).ratio() >= 0.92:
                return "near_duplicate"
        return None

    def add(self, query: str) -> None:
        normalized = _normalize_text(query)
        self._seen.add(normalized)
        self._buckets[self._bucket(query)].append(self._comparison_text(query))


@dataclass(frozen=True, slots=True)
class _FilterDecision:
    accepted: bool
    flags: tuple[str, ...]


def _filter_query(
    query: str,
    passage: CanonicalPassage,
    *,
    deduplicator: _QueryDeduplicator,
    min_source_token_overlap: float,
) -> _FilterDecision:
    flags: list[str] = []
    normalized_query = _normalize_text(query)
    if len(_tokens(query)) < 6 or _TRIVIAL_WORDS.issuperset(_tokens(query)):
        return _FilterDecision(False, ("trivial",))
    if not any(char in _VIETNAMESE_MARKERS for char in query):
        return _FilterDecision(False, ("invalid_vietnamese",))

    source_text = passage.reader_text
    normalized_source = _normalize_text(source_text)
    source_fragments = re.split(r"[.!?\n]+", normalized_source)
    if normalized_query in normalized_source or any(
        SequenceMatcher(None, normalized_query, fragment).ratio() >= 0.92
        for fragment in source_fragments
        if fragment
    ):
        return _FilterDecision(False, ("copy_source",))

    duplicate_reason = deduplicator.reason(query)
    if duplicate_reason:
        return _FilterDecision(False, (duplicate_reason,))

    citations = parse_citations(query)
    for citation in citations:
        if citation.document_number:
            return _FilterDecision(False, ("wrong_citation",))
        if citation.article:
            if (
                passage.article_number is None
                or citation.article.casefold() != passage.article_number.casefold()
            ):
                return _FilterDecision(False, ("wrong_citation",))
        if citation.clause:
            if (
                passage.clause_number is None
                or citation.clause.casefold() != passage.clause_number.casefold()
            ):
                return _FilterDecision(False, ("wrong_citation",))
    flags.append("citation_checked")

    query_tokens = _content_tokens(query)
    source_tokens = _content_tokens(source_text)
    overlap = len(query_tokens & source_tokens) / max(len(query_tokens), 1)
    if overlap < min_source_token_overlap:
        return _FilterDecision(False, ("unsupported_lexical_heuristic",))
    flags.append("heuristic_supportable")
    return _FilterDecision(True, tuple(flags))


def _synthetic_id(
    *,
    passage: CanonicalPassage,
    query_type: QueryType,
    generator: QueryGenerator,
    prompt_version: str,
) -> str:
    payload = {
        "generator_model": generator.model,
        "generator_revision": generator.revision,
        "passage_id": passage.passage_id,
        "prompt_version": prompt_version,
        "query_type": query_type,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"syn-{sha256(serialized.encode('utf-8')).hexdigest()[:24]}"


def validate_document_isolation(
    records: Sequence[SyntheticQueryRecord],
) -> tuple[bool, dict[str, tuple[str, ...]]]:
    """Verify that one source document is never assigned to multiple splits."""

    by_document: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        by_document[record.source_document_id].add(record.source_split)
    conflicts = {
        document_id: tuple(sorted(splits))
        for document_id, splits in by_document.items()
        if len(splits) > 1
    }
    return not conflicts, conflicts


def build_synthetic_records(
    passages: Sequence[CanonicalPassage],
    *,
    generator: QueryGenerator,
    config: SyntheticGenerationConfig | None = None,
    document_splits: dict[str, DocumentSplit] | None = None,
    document_split_source: Literal[
        "deterministic_hash", "external_manifest"
    ] = "deterministic_hash",
    rejected_records: list[SyntheticRejectionRecord] | None = None,
    generation_errors: list[SyntheticGenerationErrorRecord] | None = None,
) -> tuple[tuple[SyntheticQueryRecord, ...], SyntheticBuildReport]:
    """Generate accepted records from passages while enforcing document splits."""

    cfg = config or SyntheticGenerationConfig()
    ordered_passages = tuple(sorted(passages, key=lambda passage: passage.passage_id))
    passage_ids = [passage.passage_id for passage in ordered_passages]
    if len(set(passage_ids)) != len(passage_ids):
        raise SyntheticQueryError("Duplicate canonical passage IDs")
    document_ids = {passage.document_id for passage in ordered_passages}
    splits = document_splits or assign_document_splits(
        document_ids,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
    )
    if set(splits) != document_ids:
        raise SyntheticQueryError("Document split map does not cover the corpus")
    if not set(splits.values()).issubset({"train", "validation", "test"}):
        raise SyntheticQueryError("Document split map contains an unknown split")

    document_counts = Counter(splits.values())
    deduplicator = _QueryDeduplicator()
    rejection_counts: Counter[str] = Counter()
    records: list[SyntheticQueryRecord] = []
    attempts = 0
    for passage in ordered_passages:
        if splits[passage.document_id] != cfg.source_split:
            continue
        for query_type in cfg.query_types:
            if len(records) >= cfg.target_accepted or attempts >= cfg.max_attempts:
                break
            attempts += 1
            try:
                generated = generator.generate(passage, query_type)
            except Exception as exc:
                rejection_counts["generation_error"] += 1
                if generation_errors is not None:
                    generation_errors.append(
                        SyntheticGenerationErrorRecord(
                            passage_id=passage.passage_id,
                            source_document_id=passage.document_id,
                            query_type=query_type,
                            error_type=type(exc).__name__,
                            error_message=" ".join(str(exc).split())[:500]
                            or "generator failed without a message",
                        )
                    )
                continue
            candidate_query = _sanitize_generated_query(generated.query)
            decision = _filter_query(
                candidate_query,
                passage,
                deduplicator=deduplicator,
                min_source_token_overlap=cfg.min_source_token_overlap,
            )
            if not decision.accepted:
                for flag in decision.flags:
                    rejection_counts[flag] += 1
                if rejected_records is not None:
                    rejected_records.append(
                        SyntheticRejectionRecord(
                            passage_id=passage.passage_id,
                            source_document_id=passage.document_id,
                            query_type=query_type,
                            source_hash=passage.source.content_hash,
                            rejection_reason=decision.flags[0],
                        )
                    )
                continue
            deduplicator.add(candidate_query)
            records.append(
                SyntheticQueryRecord(
                    synthetic_id=_synthetic_id(
                        passage=passage,
                        query_type=query_type,
                        generator=generator,
                        prompt_version=cfg.prompt_version,
                    ),
                    query=candidate_query,
                    positive_passage_id=passage.passage_id,
                    source_document_id=passage.document_id,
                    query_type=query_type,
                    legal_domain=cfg.legal_domain,
                    generator_model=generator.model,
                    generator_revision=generator.revision,
                    prompt_version=cfg.prompt_version,
                    source_hash=passage.source.content_hash,
                    quality_flags=decision.flags,
                    raw_generation=_sanitize_generated_query(generated.raw_generation)
                    or candidate_query,
                    source_split=cfg.source_split,
                )
            )
        if len(records) >= cfg.target_accepted or attempts >= cfg.max_attempts:
            break

    isolation_ok, _ = validate_document_isolation(records)
    source_split_leakage_count = sum(
        1 for record in records if splits[record.source_document_id] != cfg.source_split
    )
    isolation_ok = isolation_ok and source_split_leakage_count == 0
    report = SyntheticBuildReport(
        schema_version=SYNTHETIC_SCHEMA_VERSION,
        attempts=attempts,
        accepted=len(records),
        target_accepted=cfg.target_accepted,
        source_document_count=len(document_ids),
        train_document_count=document_counts.get("train", 0),
        validation_document_count=document_counts.get("validation", 0),
        test_document_count=document_counts.get("test", 0),
        source_split_leakage_count=source_split_leakage_count,
        source_split=cfg.source_split,
        document_split_source=document_split_source,
        rejection_counts=dict(rejection_counts),
        generation_error_count=rejection_counts.get("generation_error", 0),
        document_isolation_ok=isolation_ok,
        prompt_version=cfg.prompt_version,
        prompt_sha256=synthetic_prompt_sha256(),
    )
    return tuple(records), report


def _write_model_jsonl(
    path: str | Path,
    records: Iterable[DomainModel],
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )


def write_synthetic_records(
    path: str | Path,
    records: Iterable[SyntheticQueryRecord],
    *,
    overwrite: bool = False,
) -> None:
    """Write accepted records without source text or answer fields."""

    _write_model_jsonl(path, records, overwrite=overwrite)


def write_synthetic_rejections(
    path: str | Path,
    records: Iterable[SyntheticRejectionRecord],
    *,
    overwrite: bool = False,
) -> None:
    """Write rejection metadata without persisting candidate generations."""

    _write_model_jsonl(path, records, overwrite=overwrite)


def write_generation_errors(
    path: str | Path,
    records: Iterable[SyntheticGenerationErrorRecord],
    *,
    overwrite: bool = False,
) -> None:
    """Write provider errors without prompts or model-output text."""

    _write_model_jsonl(path, records, overwrite=overwrite)


def load_synthetic_records(path: str | Path) -> tuple[SyntheticQueryRecord, ...]:
    """Load and validate accepted synthetic records with duplicate detection."""

    records: list[SyntheticQueryRecord] = []
    seen_ids: set[str] = set()
    for line in iter_jsonl_lines(path):
        record = SyntheticQueryRecord.model_validate_json(line)
        if record.synthetic_id in seen_ids:
            raise SyntheticQueryError(
                f"Duplicate synthetic ID in {Path(path)}: {record.synthetic_id}"
            )
        seen_ids.add(record.synthetic_id)
        records.append(record)
    return tuple(sorted(records, key=lambda record: record.synthetic_id))


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "GENERATOR_PROMPT",
    "QUERY_TYPES",
    "SYNTHETIC_SCHEMA_VERSION",
    "DocumentSplit",
    "GeneratedQuery",
    "QueryGenerator",
    "SyntheticBuildReport",
    "SyntheticGenerationConfig",
    "SyntheticGenerationErrorRecord",
    "SyntheticQueryError",
    "SyntheticRejectionRecord",
    "SyntheticQueryRecord",
    "TemplateQueryGenerator",
    "TransformersQueryGenerator",
    "assign_document_splits",
    "build_synthetic_records",
    "load_synthetic_records",
    "synthetic_prompt_sha256",
    "validate_document_isolation",
    "write_synthetic_records",
    "write_synthetic_rejections",
    "write_generation_errors",
]
