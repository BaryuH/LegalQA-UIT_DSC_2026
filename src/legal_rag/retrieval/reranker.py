"""Provider-neutral reranking contracts and deterministic offline adapters."""

from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from ..config import RerankerSection
from ..schemas import RetrievalHit

RerankScoreFunction = Callable[[str, RetrievalHit], float]
RerankMetadataValue: TypeAlias = str | int | float | bool | None
CandidateTextLookup: TypeAlias = Mapping[str, str] | Callable[[RetrievalHit], str]
SemanticModelLoader: TypeAlias = Callable[[str, str, str | None], object]

DEFAULT_SEMANTIC_RERANKER_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_LENGTH = 512


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranker query must be a non-blank string")


def _validated_hits(hits: Sequence[RetrievalHit]) -> tuple[RetrievalHit, ...]:
    normalized = tuple(hits)
    if any(not isinstance(hit, RetrievalHit) for hit in normalized):
        raise TypeError("Reranker hits must contain RetrievalHit records")
    chunk_ids = tuple(hit.chunk_id for hit in normalized)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Reranker hits must not contain duplicate chunk IDs")
    return normalized


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _validate_metadata(
    metadata: Mapping[str, RerankMetadataValue],
) -> Mapping[str, RerankMetadataValue]:
    normalized: dict[str, RerankMetadataValue] = {}
    for key, value in metadata.items():
        normalized_key = _require_non_blank(str(key), "RerankResult metadata key")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError(
                f"RerankResult metadata value for {normalized_key!r} must be scalar"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"RerankResult metadata value for {normalized_key!r} must be finite"
            )
        normalized[normalized_key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Reranked hits plus explicit usage and fallback metadata.

    ``RetrievalHit.bm25_score`` is never replaced. An adapter that is actually
    used must put its independent score in ``RetrievalHit.rerank_score``.
    When reranking is not used, ``fallback_reason`` is required so a BM25
    fallback cannot be mistaken for a successful semantic rerank.
    """

    hits: tuple[RetrievalHit, ...]
    used: bool
    model: str
    fallback_reason: str | None = None
    metadata: Mapping[str, RerankMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_hits = _validated_hits(self.hits)
        object.__setattr__(self, "hits", normalized_hits)
        if not isinstance(self.used, bool):
            raise TypeError("RerankResult.used must be a boolean")
        _require_non_blank(self.model, "RerankResult.model")
        object.__setattr__(self, "metadata", _validate_metadata(self.metadata))
        if self.fallback_reason is not None:
            _require_non_blank(
                self.fallback_reason,
                "RerankResult.fallback_reason",
            )
        if not self.used and self.fallback_reason is None:
            raise ValueError(
                "RerankResult.fallback_reason is required when reranking is not used"
            )
        if self.used and any(hit.rerank_score is None for hit in normalized_hits):
            raise ValueError(
                "Used rerank results must provide rerank_score for every hit"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize metadata and score-bearing hits without source text."""

        return {
            "hits": [hit.model_dump(mode="json") for hit in self.hits],
            "used": self.used,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class Reranker(Protocol):
    """Protocol shared by semantic rerankers and offline test adapters."""

    model: str

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        candidate_texts: Mapping[str, str] | None = None,
    ) -> RerankResult:
        """Return deterministically ordered hits for one question query."""


# Descriptive alias for callers that prefer an explicit protocol suffix.
RerankerProtocol = Reranker


class NoOpReranker:
    """Explicit BM25-only fallback used when reranking is disabled."""

    def __init__(
        self,
        *,
        model: str = "none",
        fallback_reason: str = "reranker_disabled",
    ) -> None:
        self.model = _require_non_blank(model, "NoOpReranker.model")
        self.fallback_reason = _require_non_blank(
            fallback_reason,
            "NoOpReranker.fallback_reason",
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        candidate_texts: Mapping[str, str] | None = None,
    ) -> RerankResult:
        """Preserve BM25 order and both existing score fields unchanged."""

        _validate_query(query)
        return RerankResult(
            hits=_validated_hits(hits),
            used=False,
            model=self.model,
            fallback_reason=self.fallback_reason,
        )


# Accept the spelling commonly used in configuration and issue text.
NoopReranker = NoOpReranker


class MockReranker:
    """Deterministic reranker for offline tests; never downloads a model.

    ``scores`` can provide a score by chunk ID. A ``score_fn`` may be supplied
    for query-aware fixtures. Unspecified chunk IDs receive ``1 / rank`` so the
    default behavior remains deterministic and preserves the BM25 ordering.
    """

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        model: str = "mock-reranker-v1",
        score_fn: RerankScoreFunction | None = None,
    ) -> None:
        if scores is not None and score_fn is not None:
            raise ValueError("Provide either scores or score_fn, not both")
        self.model = _require_non_blank(model, "MockReranker.model")
        self._scores = self._validate_scores(scores or {})
        self._score_fn = score_fn

    @staticmethod
    def _validate_scores(scores: Mapping[str, float]) -> dict[str, float]:
        validated: dict[str, float] = {}
        for chunk_id, score in scores.items():
            normalized_id = _require_non_blank(str(chunk_id), "MockReranker chunk ID")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise TypeError("MockReranker scores must be numeric")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError("MockReranker scores must be finite")
            validated[normalized_id] = numeric_score
        return validated

    def _score(self, query: str, hit: RetrievalHit) -> float:
        if self._score_fn is not None:
            score = self._score_fn(query, hit)
        else:
            score = self._scores.get(hit.chunk_id, 1.0 / hit.rank)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError("MockReranker score function must return a number")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("MockReranker score function must return a finite number")
        return numeric_score

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        candidate_texts: Mapping[str, str] | None = None,
    ) -> RerankResult:
        """Assign independent mock scores and rank ties by canonical chunk ID."""

        _validate_query(query)
        normalized_hits = _validated_hits(hits)
        scored = []
        for hit in normalized_hits:
            score = self._score(query, hit)
            scored.append((score, hit.model_copy(update={"rerank_score": score})))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        reranked_hits = tuple(
            hit.model_copy(update={"rank": rank})
            for rank, (_, hit) in enumerate(scored, start=1)
        )
        return RerankResult(
            hits=reranked_hits,
            used=True,
            model=self.model,
        )


class SemanticRerankerError(RuntimeError):
    """Base error for semantic-reranker configuration or execution failures."""


class SemanticRerankerUnavailableError(SemanticRerankerError):
    """Raised when a required semantic reranker cannot be used."""


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unresolved"


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Reranker device must be one of: auto, cpu, cuda")
    if device == "cpu":
        return "cpu"

    try:
        torch = import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        if device == "cuda":
            raise SemanticRerankerUnavailableError("cuda_unavailable") from exc
        return "cpu"

    if device == "cuda" and not cuda_available:
        raise SemanticRerankerUnavailableError("cuda_unavailable")
    return "cuda" if cuda_available else "cpu"


def _default_model_loader(
    model_name: str,
    device: str,
    model_revision: str | None,
) -> object:
    """Load SentenceTransformers lazily so importing the package stays offline."""

    sentence_transformers = import_module("sentence_transformers")
    sentence_transformer = sentence_transformers.SentenceTransformer
    kwargs: dict[str, str] = {"device": device}
    if model_revision is not None:
        kwargs["revision"] = model_revision
    return sentence_transformer(model_name, **kwargs)


def _as_vector_rows(encoded: object) -> tuple[tuple[float, ...], ...]:
    value = encoded
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    try:
        rows = list(cast(Iterable[object], value))
    except TypeError as exc:
        raise SemanticRerankerError("Encoder output must be a 2D sequence") from exc
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]

    normalized: list[tuple[float, ...]] = []
    for row in rows:
        try:
            values = tuple(
                float(cast(Any, item)) for item in cast(Iterable[object], row)
            )
        except (TypeError, ValueError) as exc:
            raise SemanticRerankerError(
                "Encoder output must contain numeric vectors"
            ) from exc
        if not values or any(not math.isfinite(item) for item in values):
            raise SemanticRerankerError("Encoder output vectors must be finite")
        normalized.append(values)
    return tuple(normalized)


def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise SemanticRerankerError("Encoder output vector dimensions do not match")
    dot = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    score = dot / (first_norm * second_norm)
    if not math.isfinite(score):
        raise SemanticRerankerError("Cosine score must be finite")
    return score


class SemanticReranker:
    """Optional SentenceTransformers reranker over BM25 candidate chunks.

    The adapter accepts candidate text separately from ``RetrievalHit`` because
    hits intentionally contain provenance and scores, not legal passage text.
    ``candidate_texts`` is therefore a derived, question-safe mapping keyed by
    chunk ID. The source chunk and BM25 hit are never mutated.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_RERANKER_MODEL,
        *,
        required: bool = False,
        device: str = "auto",
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        model_revision: str | None = None,
        encoder: object | None = None,
        model_loader: SemanticModelLoader | None = None,
        text_lookup: CandidateTextLookup | None = None,
    ) -> None:
        self.model = _require_non_blank(model_name, "SemanticReranker.model")
        if not isinstance(required, bool):
            raise TypeError("SemanticReranker.required must be a boolean")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("SemanticReranker.batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("SemanticReranker.batch_size must be greater than zero")
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("SemanticReranker.max_length must be an integer")
        if max_length <= 0:
            raise ValueError("SemanticReranker.max_length must be greater than zero")
        if model_revision is not None:
            _require_non_blank(
                model_revision,
                "SemanticReranker.model_revision",
            )

        self.required = required
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_revision = model_revision
        self._text_lookup = text_lookup
        self._library_version = _package_version("sentence-transformers")
        self._encoder: object | None = None
        self._load_error_reason: str | None = None
        self._resolved_device = device
        self._model_load_latency_ms = 0.0
        self._last_report: dict[str, RerankMetadataValue] = {}

        load_started = time.perf_counter()
        try:
            self._resolved_device = _resolve_device(device)
            selected_loader = model_loader or _default_model_loader
            self._encoder = (
                encoder
                if encoder is not None
                else selected_loader(
                    self.model,
                    self._resolved_device,
                    model_revision,
                )
            )
            if not callable(getattr(self._encoder, "encode", None)):
                raise TypeError("Semantic encoder must expose encode()")
            self._configure_encoder()
        except Exception as exc:
            detail = (
                str(exc)
                if isinstance(exc, SemanticRerankerError)
                else type(exc).__name__
            )
            self._load_error_reason = f"semantic_reranker_unavailable:{detail}"
            self._encoder = None
            if required:
                raise SemanticRerankerUnavailableError(self._load_error_reason) from exc
        finally:
            self._model_load_latency_ms = _milliseconds_since(load_started)

    @property
    def resolved_device(self) -> str:
        """Return the concrete CPU/CUDA device selected for this instance."""

        return self._resolved_device

    @property
    def last_report(self) -> dict[str, RerankMetadataValue]:
        """Return the latest content-free latency/memory report."""

        return dict(self._last_report)

    def _configure_encoder(self) -> None:
        if self._encoder is None:  # pragma: no cover - guarded by __init__
            raise SemanticRerankerUnavailableError("encoder_missing")
        try:
            cast(Any, self._encoder).max_seq_length = self.max_length
        except Exception:
            # The explicit text truncation path below remains the safety boundary.
            pass

    def _resolve_candidate_texts(
        self,
        hits: Sequence[RetrievalHit],
        candidate_texts: Mapping[str, str] | None,
    ) -> tuple[str, ...]:
        lookup = candidate_texts if candidate_texts is not None else self._text_lookup
        if lookup is None:
            raise ValueError(
                "Semantic reranking requires candidate_texts keyed by chunk ID"
            )

        resolved: list[str] = []
        for hit in hits:
            value = (
                lookup.get(hit.chunk_id) if isinstance(lookup, Mapping) else lookup(hit)
            )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Missing non-blank candidate text for chunk {hit.chunk_id!r}"
                )
            resolved.append(value)
        return tuple(resolved)

    def _truncate_text(self, text: str) -> tuple[str, str]:
        if self._encoder is not None:
            tokenizer = getattr(self._encoder, "tokenizer", None)
            encode = getattr(tokenizer, "encode", None)
            decode = getattr(tokenizer, "decode", None)
            if callable(encode) and callable(decode):
                try:
                    token_ids = encode(
                        text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    tolist = getattr(token_ids, "tolist", None)
                    if callable(tolist):
                        token_ids = tolist()
                    return (
                        str(decode(token_ids, skip_special_tokens=True)),
                        "tokenizer",
                    )
                except Exception:
                    pass

        # SentenceTransformers normally exposes a tokenizer. This bounded
        # character approximation is a final safety boundary for test doubles
        # and unusual compatible encoders; source text is never overwritten.
        char_limit = max(1, self.max_length * 4)
        return text[:char_limit].rstrip(), "character_approximation"

    def _encode(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[tuple[float, ...], ...], float, float]:
        if self._encoder is None:
            raise SemanticRerankerUnavailableError("encoder_missing")
        tracing_before = tracemalloc.is_tracing()
        if not tracing_before:
            tracemalloc.start()
        started = time.perf_counter()
        try:
            encoded = cast(Any, self._encoder).encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = _as_vector_rows(encoded)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            if not tracing_before:
                tracemalloc.stop()
        return vectors, _milliseconds_since(started), peak_bytes / (1024 * 1024)

    def _metadata(
        self,
        *,
        input_count: int,
        latency_ms: float,
        peak_memory_mb: float,
        truncation: str,
    ) -> dict[str, RerankMetadataValue]:
        return {
            "provider": "sentence_transformers",
            "model_name": self.model,
            "model_version": self.model_revision or "unresolved",
            "library_version": self._library_version,
            "requested_device": self.requested_device,
            "device": self.resolved_device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "input_count": input_count,
            "truncation": truncation,
            "latency_ms": round(latency_ms, 3),
            "peak_memory_mb": round(peak_memory_mb, 3),
            "memory_measurement": "tracemalloc_python_allocations",
            "model_load_latency_ms": round(self._model_load_latency_ms, 3),
        }

    def _fallback(
        self,
        hits: tuple[RetrievalHit, ...],
        reason: str,
        *,
        latency_ms: float = 0.0,
        peak_memory_mb: float = 0.0,
        input_count: int = 0,
        truncation: str = "not_run",
    ) -> RerankResult:
        metadata = self._metadata(
            input_count=input_count,
            latency_ms=latency_ms,
            peak_memory_mb=peak_memory_mb,
            truncation=truncation,
        )
        self._last_report = metadata
        return RerankResult(
            hits=hits,
            used=False,
            model=self.model,
            fallback_reason=reason,
            metadata=metadata,
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        candidate_texts: Mapping[str, str] | None = None,
    ) -> RerankResult:
        """Rerank BM25 hits, or return an explicit BM25 fallback."""

        _validate_query(query)
        normalized_hits = _validated_hits(hits)
        if not normalized_hits:
            metadata = self._metadata(
                input_count=0,
                latency_ms=0.0,
                peak_memory_mb=0.0,
                truncation="not_run",
            )
            self._last_report = metadata
            return RerankResult(
                hits=(),
                used=self._load_error_reason is None,
                model=self.model,
                fallback_reason=self._load_error_reason,
                metadata=metadata,
            )

        if self._load_error_reason is not None:
            return self._fallback(
                normalized_hits,
                self._load_error_reason,
                input_count=len(normalized_hits) + 1,
            )

        candidate_text_values = self._resolve_candidate_texts(
            normalized_hits,
            candidate_texts,
        )

        prepared_texts = [self._truncate_text(query)]
        prepared_texts.extend(
            self._truncate_text(text) for text in candidate_text_values
        )
        truncation_modes = {mode for _, mode in prepared_texts}
        truncation = (
            next(iter(truncation_modes)) if len(truncation_modes) == 1 else "mixed"
        )
        texts = tuple(text for text, _ in prepared_texts)
        encode_started = time.perf_counter()
        try:
            vectors, latency_ms, peak_memory_mb = self._encode(texts)
            if len(vectors) != len(texts):
                raise SemanticRerankerError(
                    "Encoder output count does not match input count"
                )
            query_vector = vectors[0]
            scores = tuple(
                _cosine_similarity(query_vector, vector) for vector in vectors[1:]
            )
        except Exception as exc:
            reason = f"semantic_reranker_failed:{type(exc).__name__}"
            if self.required:
                raise SemanticRerankerUnavailableError(reason) from exc
            return self._fallback(
                normalized_hits,
                reason,
                latency_ms=_milliseconds_since(encode_started),
                input_count=len(texts),
                truncation=truncation,
            )

        scored = [
            (
                score,
                hit.model_copy(update={"rerank_score": score}),
            )
            for score, hit in zip(scores, normalized_hits, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        reranked_hits = tuple(
            hit.model_copy(update={"rank": rank})
            for rank, (_, hit) in enumerate(scored, start=1)
        )
        metadata = self._metadata(
            input_count=len(texts),
            latency_ms=latency_ms,
            peak_memory_mb=peak_memory_mb,
            truncation=truncation,
        )
        self._last_report = metadata
        return RerankResult(
            hits=reranked_hits,
            used=True,
            model=self.model,
            metadata=metadata,
        )


def _milliseconds_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


# Explicit name for callers that prefer the provider implementation name.
SentenceTransformerReranker = SemanticReranker


def create_reranker(
    config: RerankerSection,
    *,
    encoder: object | None = None,
    model_loader: SemanticModelLoader | None = None,
    text_lookup: CandidateTextLookup | None = None,
) -> Reranker:
    """Create the configured reranker without importing model dependencies eagerly."""

    if not config.enabled or config.provider == "none":
        return NoOpReranker()
    if config.provider == "mock":
        return MockReranker(model=config.model or "mock-reranker-v1")
    if config.provider == "sentence_transformers":
        return SemanticReranker(
            model_name=config.model or DEFAULT_SEMANTIC_RERANKER_MODEL,
            model_revision=config.model_revision,
            required=config.required,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            encoder=encoder,
            model_loader=model_loader,
            text_lookup=text_lookup,
        )
    raise ValueError(f"Unsupported reranker provider: {config.provider!r}")


__all__ = [
    "CandidateTextLookup",
    "DEFAULT_RERANKER_BATCH_SIZE",
    "DEFAULT_RERANKER_MAX_LENGTH",
    "DEFAULT_SEMANTIC_RERANKER_MODEL",
    "MockReranker",
    "NoOpReranker",
    "NoopReranker",
    "RerankResult",
    "RerankMetadataValue",
    "RerankScoreFunction",
    "Reranker",
    "RerankerProtocol",
    "SemanticModelLoader",
    "SemanticReranker",
    "SemanticRerankerError",
    "SemanticRerankerUnavailableError",
    "SentenceTransformerReranker",
    "create_reranker",
]
