"""Cross-encoder reranking with ``AITeamVN/Vietnamese_Reranker`` (TASK 24).

Why this model, and why a separate module from ``cross_encoder.py``.

**Why a reranker at all.** On clean-460 the candidate set holds the right
article for 97.45% of labeled queries at rank 500 and 88.32% at rank 10, but the
top-4 the reader receives holds it for 79.20%. Recall is near its ceiling;
nothing in the current path reads a (query, passage) pair jointly. Published
deltas on this exact task family: the DRiLL@VLSP 2025 organiser baseline goes
from **F2 0.3365 (BM25) to 0.5512 (BM25 + fine-tuned cross-encoder)**; ViDRILL's
ablation puts reranking at **+0.13 to +0.15 F2**, larger than any index change
they tried.

**Why this checkpoint.** On MMARCO-VI, the only public Vietnamese reranking
benchmark, the Vietnamese-specific rerankers beat ``bge-reranker-v2-m3`` by 5-7
NDCG points (ViRanker NDCG@3 0.6815, PhoRanker NDCG@10 0.7422, bge-reranker-v2-m3
0.6087 / 0.6872). PhoRanker is disqualified here by a **256-token total limit**
against a corpus whose median unit is 932 characters and p90 is 2,210.
``AITeamVN/Vietnamese_Reranker`` takes **query 256 + passage 2048 tokens**, is
Apache-2.0, is a ``bge-reranker-v2-m3`` fine-tune on ~1.1M Vietnamese triplets,
and is the only shortlisted model with a published Vietnamese *legal* number:
on Legal Zalo 2021 it lifts **Acc@1 0.7274 -> 0.7944 and MRR@10 0.8181 -> 0.8672**
over its own retriever.

**Why not reuse ``CrossEncoderScorer``.** That adapter goes through
``sentence_transformers.CrossEncoder``, which applies a sigmoid when the model
has one label. ``cross_encoder.py`` documents the consequence itself: an
irrelevant passage scores 0.000 and a relevant one 0.997, so much of a top-100
tail ties exactly. A saturated score cannot carry a threshold, and the threshold
is the point - the largest single number in the DRiLL evidence is **+0.308 F2
from replacing a fixed top-k cut with a variable-size answer set**, which needs a
score with usable spread. This module therefore reads the model's raw logits
through ``transformers`` directly, and truncates the query and the passage
against their own budgets so a long article can never crowd out the question.

Everything except :class:`VietnameseRerankerScorer` is a pure function of a
score callable and is testable without weights.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

VIETNAMESE_RERANKER_SCHEMA_VERSION = "sedar-vietnamese-reranker-v1"

#: Pinned defaults from the model card. Changing these is a model change and
#: must be recorded in the run artifact, not edited silently.
DEFAULT_MODEL = "AITeamVN/Vietnamese_Reranker"
DEFAULT_MAX_QUERY_TOKENS = 256
DEFAULT_MAX_PASSAGE_TOKENS = 2048
DEFAULT_MAX_LENGTH = 2304

ScoreMode = Literal["raw_logit", "sigmoid"]
TextField = Literal["reader_text", "raw_text", "breadcrumb_reader_text"]

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MAX_PASSAGE_TOKENS",
    "DEFAULT_MAX_QUERY_TOKENS",
    "DEFAULT_MODEL",
    "CutoffPolicy",
    "CutoffResult",
    "LoadedUnit",
    "load_rerank_units",
    "RerankUnit",
    "RerankedCandidate",
    "VIETNAMESE_RERANKER_SCHEMA_VERSION",
    "VietnameseRerankerConfig",
    "VietnameseRerankerError",
    "VietnameseRerankerScorer",
    "apply_cutoff",
    "merge_children_to_parent",
    "rerank_query",
    "unit_text",
]


class VietnameseRerankerError(RuntimeError):
    """Raised when reranking cannot proceed without violating the contract."""


class RerankUnit(Protocol):
    """The subset of a corpus unit this module needs.

    Satisfied by ``corpus_v4.RetrievalUnit`` directly and by a thin adapter over
    the v3 ``CanonicalPassage`` (whose ``parent_unit_id`` is simply ``None``).
    """

    unit_id: str
    document_id: str
    parent_unit_id: str | None
    level: str
    role: str
    reader_text: str
    breadcrumb: str
    char_count: int
    packable: bool


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """One candidate after cross-encoder scoring."""

    unit_id: str
    document_id: str
    score: float
    rank: int
    retriever_rank: int
    #: Set when this candidate's score was carried up from a child unit.
    merged_from: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "ce_score": self.score,
            "ce_rank": self.rank,
            "retriever_rank": self.retriever_rank,
            "merged_from": list(self.merged_from),
        }


def unit_text(unit: RerankUnit, *, text_field: TextField = "reader_text") -> str:
    """Return the passage side of the pair handed to the cross-encoder.

    ``reader_text`` is the default because it already opens with the article
    heading (``Điều 76. ...``) and carries no index-only decoration.
    ``breadcrumb_reader_text`` prepends the one-line hierarchy path, which the
    2,048-token passage budget can afford and which disambiguates units whose
    body is boilerplate shared across documents. It is an ablation arm, not a
    default: the frozen champion's evidence distribution does not include it.
    """

    if text_field == "raw_text":
        return getattr(unit, "raw_text", unit.reader_text)
    if text_field == "breadcrumb_reader_text":
        breadcrumb = (unit.breadcrumb or "").strip()
        body = unit.reader_text.strip()
        return f"{breadcrumb}\n{body}".strip() if breadcrumb else body
    return unit.reader_text


def merge_children_to_parent(
    scored: Sequence[RerankedCandidate],
    units: Mapping[str, RerankUnit],
    *,
    aggregate: Literal["max", "mean"] = "max",
) -> tuple[RerankedCandidate, ...]:
    """Collapse ``article_part`` hits onto their parent ``Điều``.

    Corpus v4 indexes a long article through merged clause children and keeps the
    article as the scoring unit, so a child hit must be scored as its parent
    before any cutoff is applied - otherwise one article occupies several slots
    and the pack starves, which is the defect recorded in
    ``memory-bank/progress.md``.

    ``max`` is the default rather than ``sum``: summing rewards an article merely
    for having been split into many pieces. This is the auto-merge rule from
    hierarchical retrieval, with the parent chosen by structure rather than by a
    fraction-of-children threshold, because here the structure is authoritative.
    """

    if aggregate not in {"max", "mean"}:
        raise VietnameseRerankerError(f"Unknown aggregate: {aggregate!r}")

    best: dict[str, RerankedCandidate] = {}
    contributors: dict[str, list[str]] = {}
    totals: dict[str, list[float]] = {}

    for candidate in scored:
        unit = units.get(candidate.unit_id)
        if unit is None:
            raise VietnameseRerankerError(
                f"Unit {candidate.unit_id!r} is absent from the corpus view"
            )
        parent_id = unit.parent_unit_id or candidate.unit_id
        if parent_id != candidate.unit_id and parent_id not in units:
            # A parent that is not in the view cannot be emitted; keep the child.
            parent_id = candidate.unit_id
        parent_unit = units.get(parent_id, unit)
        totals.setdefault(parent_id, []).append(candidate.score)
        if parent_id != candidate.unit_id:
            contributors.setdefault(parent_id, []).append(candidate.unit_id)
        current = best.get(parent_id)
        if current is None or candidate.score > current.score:
            best[parent_id] = RerankedCandidate(
                unit_id=parent_id,
                document_id=parent_unit.document_id,
                score=candidate.score,
                rank=0,
                retriever_rank=candidate.retriever_rank
                if current is None
                else min(current.retriever_rank, candidate.retriever_rank),
            )
        elif candidate.retriever_rank < current.retriever_rank:
            best[parent_id] = RerankedCandidate(
                unit_id=current.unit_id,
                document_id=current.document_id,
                score=current.score,
                rank=0,
                retriever_rank=candidate.retriever_rank,
            )

    merged: list[RerankedCandidate] = []
    for parent_id, candidate in best.items():
        score = candidate.score
        if aggregate == "mean":
            values = totals[parent_id]
            score = sum(values) / len(values)
        merged.append(
            RerankedCandidate(
                unit_id=parent_id,
                document_id=candidate.document_id,
                score=score,
                rank=0,
                retriever_rank=candidate.retriever_rank,
                merged_from=tuple(sorted(set(contributors.get(parent_id, ())))),
            )
        )

    # Stable, fully specified order: score desc, then the retriever's own rank,
    # then the id. Never break a tie on the id alone - that would scramble the
    # tail into id order and discard the retriever's ranking, which is the only
    # signal left where the cross-encoder has stopped discriminating.
    merged.sort(key=lambda item: (-item.score, item.retriever_rank, item.unit_id))
    return tuple(
        RerankedCandidate(
            unit_id=item.unit_id,
            document_id=item.document_id,
            score=item.score,
            rank=index,
            retriever_rank=item.retriever_rank,
            merged_from=item.merged_from,
        )
        for index, item in enumerate(merged, start=1)
    )


@dataclass(frozen=True, slots=True)
class CutoffPolicy:
    """Variable-size answer set instead of a constant ``evidence_top_k``.

    The largest measured effect in the DRiLL@VLSP 2025 evidence is not a model:
    replacing a fixed top-10 cut with a variable-size answer set moved F2 from
    0.3636 to 0.6714 (+0.308), and the LLM stage added 0.055 on top. With 1.34
    gold articles per query, returning a constant k caps precision arithmetically
    however good the ranking is. Our own gold answers cite **1.65 distinct Điều
    on average and 38% cite two or more**, so the right pack size varies per
    query.

    ``score_threshold=None`` disables thresholding and reproduces fixed-``k``
    behaviour exactly, which is the default so that no run changes silently. It
    has to be calibrated on the clean-460 manifest before promotion; a threshold
    read off another corpus does not transfer.

    ``max_total_chars`` and ``max_per_document`` are the budget the threshold
    operates inside. They are listed after the threshold on purpose: a count cap
    that binds before the character budget is the ``evidence_top_k`` defect.
    """

    score_threshold: float | None = None
    #: Relative fallback: keep candidates within this much of the top score.
    relative_margin: float | None = None
    min_keep: int = 1
    max_keep: int = 8
    max_total_chars: int = 6000
    max_per_document: int = 3
    #: Units marked ``packable=False`` in the corpus (enforcement articles,
    #: oversized annex containers) are ranked but never consume pack budget.
    respect_packable: bool = True

    def __post_init__(self) -> None:
        if self.min_keep <= 0:
            raise ValueError("min_keep must be positive")
        if self.max_keep < self.min_keep:
            raise ValueError("max_keep must be >= min_keep")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        if self.max_per_document <= 0:
            raise ValueError("max_per_document must be positive")
        if self.relative_margin is not None and self.relative_margin < 0.0:
            raise ValueError("relative_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class CutoffResult:
    """What the cutoff kept, and why it stopped."""

    kept: tuple[RerankedCandidate, ...]
    total_chars: int
    stop_reason: str
    dropped_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept_ids": [item.unit_id for item in self.kept],
            "kept_count": len(self.kept),
            "total_chars": self.total_chars,
            "stop_reason": self.stop_reason,
            "dropped_reasons": dict(self.dropped_reasons),
        }


def apply_cutoff(
    ranked: Sequence[RerankedCandidate],
    units: Mapping[str, RerankUnit],
    policy: CutoffPolicy,
) -> CutoffResult:
    """Select a variable-size answer set under a character budget.

    Order of operations, and it matters: the score gate decides *eligibility*,
    the budget decides *how many fit*, and ``min_keep`` overrides both so a
    strict threshold can never return an empty pack.
    """

    if not ranked:
        return CutoffResult(kept=(), total_chars=0, stop_reason="no_candidates")

    top_score = ranked[0].score
    floor: float | None = None
    if policy.score_threshold is not None:
        floor = policy.score_threshold
    if policy.relative_margin is not None:
        relative_floor = top_score - policy.relative_margin
        floor = relative_floor if floor is None else max(floor, relative_floor)

    kept: list[RerankedCandidate] = []
    per_document: dict[str, int] = {}
    total_chars = 0
    dropped: dict[str, int] = {}
    stop_reason = "exhausted_candidates"

    def note(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for candidate in ranked:
        if len(kept) >= policy.max_keep:
            stop_reason = "max_keep"
            break
        unit = units.get(candidate.unit_id)
        if unit is None:
            raise VietnameseRerankerError(
                f"Unit {candidate.unit_id!r} is absent from the corpus view"
            )
        below_floor = floor is not None and candidate.score < floor
        if below_floor and len(kept) >= policy.min_keep:
            stop_reason = "score_threshold"
            break
        if policy.respect_packable and not unit.packable:
            # Unconditional: an enforcement article or an oversized annex
            # container must never enter the pack. If every candidate is
            # unpackable the empty-pack fallback below still returns the head,
            # reported as ``min_keep_override`` rather than silently.
            note("not_packable")
            continue
        if per_document.get(unit.document_id, 0) >= policy.max_per_document:
            note("max_per_document")
            continue
        size = unit.char_count or len(unit.reader_text)
        if total_chars + size > policy.max_total_chars and len(kept) >= policy.min_keep:
            note("max_total_chars")
            continue
        kept.append(
            RerankedCandidate(
                unit_id=candidate.unit_id,
                document_id=candidate.document_id,
                score=candidate.score,
                rank=len(kept) + 1,
                retriever_rank=candidate.retriever_rank,
                merged_from=candidate.merged_from,
            )
        )
        per_document[unit.document_id] = per_document.get(unit.document_id, 0) + 1
        total_chars += size

    if not kept:
        # min_keep is a floor, not a suggestion: an empty pack is worse than an
        # over-budget one, and a silently empty pack is worst of all.
        head = ranked[0]
        unit = units[head.unit_id]
        kept.append(
            RerankedCandidate(
                unit_id=head.unit_id,
                document_id=head.document_id,
                score=head.score,
                rank=1,
                retriever_rank=head.retriever_rank,
                merged_from=head.merged_from,
            )
        )
        total_chars = unit.char_count or len(unit.reader_text)
        stop_reason = "min_keep_override"

    return CutoffResult(
        kept=tuple(kept),
        total_chars=total_chars,
        stop_reason=stop_reason,
        dropped_reasons=dropped,
    )


def rerank_query(
    query: str,
    candidate_ids: Sequence[str],
    units: Mapping[str, RerankUnit],
    *,
    score_fn: Callable[[str, Sequence[str]], Sequence[float]],
    top_k: int = 100,
    text_field: TextField = "reader_text",
    aggregate: Literal["max", "mean"] = "max",
) -> tuple[RerankedCandidate, ...]:
    """Score the head of one candidate list and merge children onto parents.

    Candidates past ``top_k`` are appended below the reranked head in their
    original order rather than dropped, so recall at deeper cutoffs is unchanged
    and the output stays a drop-in replacement for the input ranking.
    """

    if top_k <= 0:
        raise VietnameseRerankerError("top_k must be positive")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise VietnameseRerankerError("Duplicate candidate id in the ranking")

    head = list(candidate_ids[:top_k])
    tail = list(candidate_ids[top_k:])
    texts: list[str] = []
    for unit_id in head:
        unit = units.get(unit_id)
        if unit is None:
            raise VietnameseRerankerError(
                f"Unit {unit_id!r} is absent from the corpus view"
            )
        texts.append(unit_text(unit, text_field=text_field))

    scores = list(score_fn(query, texts)) if texts else []
    if len(scores) != len(head):
        raise VietnameseRerankerError(
            f"Scorer returned {len(scores)} scores for {len(head)} candidates"
        )

    scored = [
        RerankedCandidate(
            unit_id=unit_id,
            document_id=units[unit_id].document_id,
            score=float(score),
            rank=0,
            retriever_rank=index,
        )
        for index, (unit_id, score) in enumerate(zip(head, scores, strict=True), start=1)
    ]
    merged = list(merge_children_to_parent(scored, units, aggregate=aggregate))

    # The tail keeps retriever order and sits strictly below every reranked
    # candidate. It is never rescored, so it must not be interleaved by score.
    seen = {item.unit_id for item in merged}
    next_rank = len(merged) + 1
    for offset, unit_id in enumerate(tail):
        if unit_id in seen:
            continue
        unit = units.get(unit_id)
        if unit is None:
            continue
        merged.append(
            RerankedCandidate(
                unit_id=unit_id,
                document_id=unit.document_id,
                score=float("-inf"),
                rank=next_rank,
                retriever_rank=top_k + offset + 1,
            )
        )
        seen.add(unit_id)
        next_rank += 1
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class VietnameseRerankerConfig:
    """Runtime configuration, pinned to the model card's own limits."""

    model: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS
    max_passage_tokens: int = DEFAULT_MAX_PASSAGE_TOKENS
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 16
    score_mode: ScoreMode = "raw_logit"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for name in ("max_query_tokens", "max_passage_tokens", "max_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_query_tokens + self.max_passage_tokens > self.max_length:
            raise ValueError(
                "max_query_tokens + max_passage_tokens must fit in max_length; "
                f"the model card gives 256 + 2048 = {DEFAULT_MAX_LENGTH}"
            )
        if self.score_mode not in {"raw_logit", "sigmoid"}:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "max_query_tokens": self.max_query_tokens,
            "max_passage_tokens": self.max_passage_tokens,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "score_mode": self.score_mode,
            "schema_version": VIETNAMESE_RERANKER_SCHEMA_VERSION,
        }


class VietnameseRerankerScorer:
    """Raw-logit adapter over ``AITeamVN/Vietnamese_Reranker``.

    Reads the model through ``transformers`` rather than
    ``sentence_transformers.CrossEncoder`` so the score is the model's own logit,
    not a sigmoid of it. The query and the passage are truncated against separate
    budgets (256 / 2048 tokens), so a 6,000-character article can never push the
    question out of the window - which pair-level ``truncation=True`` can do
    whenever the question is the longer side after tokenisation.
    """

    def __init__(
        self,
        config: VietnameseRerankerConfig | None = None,
        *,
        model_loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or VietnameseRerankerConfig()
        if model_loader is None or tokenizer_loader is None:
            try:
                from transformers import (  # noqa: PLC0415
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise VietnameseRerankerError(
                    "Vietnamese reranking requires transformers."
                ) from exc
            model_loader = model_loader or AutoModelForSequenceClassification.from_pretrained
            tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained

        kwargs: dict[str, Any] = {"local_files_only": self.config.local_files_only}
        # A local snapshot directory is read straight from disk; passing a
        # revision alongside it sends the loader back through hub resolution,
        # which is how the Qwen snapshot blocked the dense rebuild.
        if self.config.revision and not _looks_like_path(self.config.model):
            kwargs["revision"] = self.config.revision

        self.tokenizer = tokenizer_loader(self.config.model, **kwargs)
        self.model = model_loader(self.config.model, **kwargs)
        self._torch = _maybe_torch()
        if self._torch is not None:
            if self.config.device.startswith("cuda") and not self._torch.cuda.is_available():
                raise VietnameseRerankerError(
                    "device='cuda' was requested but CUDA is not available. "
                    "This runner is fail-closed: it will not silently score on CPU."
                )
            dtype = getattr(self._torch, self.config.dtype, None)
            if dtype is not None and self.config.device.startswith("cuda"):
                self.model = self.model.to(dtype=dtype)
            self.model = self.model.to(self.config.device)
            self.model.eval()

    def _truncate_query(self, query: str) -> str:
        ids = self.tokenizer(
            query,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_query_tokens,
        )["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        if not texts:
            return ()
        clipped_query = self._truncate_query(query)
        scores: list[float] = []
        for batch in _batched(texts, self.config.batch_size):
            pairs = [(clipped_query, text) for text in batch]
            encoded = self.tokenizer(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                padding=True,
                # only_second keeps the whole (already clipped) query and spends
                # the remaining budget on the passage.
                truncation="only_second",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            scores.extend(self._forward(encoded))
        return scores

    def _forward(self, encoded: Any) -> list[float]:
        torch = self._torch
        if torch is None:  # pragma: no cover - exercised only without torch
            raise VietnameseRerankerError("Scoring requires torch.")
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded, return_dict=True).logits.view(-1).float()
            if self.config.score_mode == "sigmoid":
                logits = torch.sigmoid(logits)
        return [float(value) for value in logits.cpu()]


@dataclass(slots=True)
class LoadedUnit:
    """Concrete :class:`RerankUnit` used when units are read from JSONL.

    Accepts both the corpus v4 ``units.jsonl`` shape and the v3
    ``passages_*.jsonl`` shape, so the reranker can be evaluated against either
    corpus version without a migration. A v3 passage has no ``parent_unit_id``,
    so child merging becomes a no-op there rather than an error.
    """

    unit_id: str
    document_id: str
    reader_text: str
    breadcrumb: str = ""
    parent_unit_id: str | None = None
    level: str = "article"
    role: str = "substantive"
    char_count: int = 0
    packable: bool = True
    raw_text: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "LoadedUnit":
        unit_id = str(row.get("unit_id") or row.get("passage_id") or "").strip()
        if not unit_id:
            raise VietnameseRerankerError(
                "Corpus row has neither unit_id nor passage_id"
            )
        reader_text = str(row.get("reader_text") or row.get("raw_text") or "")
        if not reader_text.strip():
            raise VietnameseRerankerError(f"Unit {unit_id!r} has no reader text")
        level = str(row.get("level") or row.get("retrieval_level") or "article")
        return cls(
            unit_id=unit_id,
            document_id=str(row.get("document_id") or ""),
            reader_text=reader_text,
            breadcrumb=str(row.get("breadcrumb") or ""),
            parent_unit_id=(str(row["parent_unit_id"]) if row.get("parent_unit_id") else None),
            level=level,
            role=str(row.get("role") or "substantive"),
            char_count=int(row.get("char_count") or len(reader_text)),
            # v3 passages carry no packability flag; treat them as packable.
            packable=bool(row.get("packable", True)),
            raw_text=str(row.get("raw_text") or ""),
        )


def load_rerank_units(path: Any) -> dict[str, LoadedUnit]:
    """Read a corpus v4 ``units.jsonl`` or a v3 passages JSONL."""

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    units: dict[str, LoadedUnit] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            unit = LoadedUnit.from_row(json.loads(line))
            if unit.unit_id in units:
                raise VietnameseRerankerError(f"Duplicate unit id {unit.unit_id!r}")
            units[unit.unit_id] = unit
    if not units:
        raise VietnameseRerankerError(f"No units found in {path}")
    return units


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _maybe_torch() -> Any | None:
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError:  # pragma: no cover - environment dependent
        return None


def _looks_like_path(model: str) -> bool:
    from pathlib import Path  # noqa: PLC0415

    try:
        return Path(model).expanduser().is_dir()
    except OSError:  # pragma: no cover - defensive
        return False
