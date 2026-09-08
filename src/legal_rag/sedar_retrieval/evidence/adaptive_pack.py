"""Adaptive evidence-pack selection (TASK 25).

This module decides *which* candidates and *how many* reach the reader. It runs
before ``pack_passage_retrieval_evidence``, so ``pack_evidence`` remains the
single place that renders blocks and enforces the final budget; everything here
is a pure function over (candidates, scores, unit metadata).

## The two findings this has to reconcile

They point in opposite directions, and getting the direction wrong is how a pack
policy makes things worse.

**Cut, says the retrieval literature.** The largest measured effect in the
DRiLL@VLSP 2025 evidence is not a model: replacing a fixed top-10 cut with a
variable-size answer set moved F2 from 0.3636 to 0.6714 (+0.308), and the LLM
stage added only 0.055 on top. With 1.34 gold articles per query, a constant k
caps precision arithmetically. "The Power of Noise" (SIGIR 2024) measures the
cost of the extra blocks directly: one semantically related but answer-free
distractor cost -25% accuracy, several up to -67%. "Lost in the Middle"
(TACL 2024) measures the diminishing return of width: 20 -> 30 documents bought
~1.5 points for GPT-3.5 and ~1% for Claude.

**Fill, says our own data.** ``memory-bank/progress.md`` records the opposite
bottleneck: *"bottom quartile is starved packs, not any coded error"*, a
*"monotone dose-response between evidence characters and METEOR"*, and
*"65 UNDER_SPECIFIED cases blamed on the generation cap are actually out of
content, not out of budget"*. The champion moved from 4/4000/2 to 6/6000/3 for
+0.0492 METEOR, and the finding was explicit that ``evidence_top_k`` alone is
inert while characters and per-document count are the real levers.

**Both are true, because the metrics differ.** Macro-F2 over cited articles
punishes over-retrieval; METEOR/ROUGE-L over generated prose punishes an answer
that had nothing to ground itself in. Task 2 is scored on the prose. So the
policy here is *not* "cut to the confident few" - it is **fill toward a
character target, gated on relevance, and say which of the two ran out.**

That last part is the point. A pack can end for two completely different
reasons, and the existing config cannot tell them apart:

``budget_exhausted``
    the character cap bound - more evidence existed and was refused.
``content_exhausted``
    every candidate was used and the pack still came out meaningfully under
    target - the retriever, not the budget, is the constraint. "Meaningfully"
    is ``starvation_ratio`` (default 0.75 of ``target_chars``), because a pack
    almost never lands exactly on its target and the measured symptom was
    specific: a quarter of cases filled less than half the budget.

Those two call for opposite fixes, and the 65 mislabelled UNDER_SPECIFIED cases
are what happens when a report cannot distinguish them.

## What the policy does

1. **Relevance gate** (``score_floor`` / ``relative_margin``) decides which
   candidates are eligible.
2. **Fill to ``target_chars``** under the hard ``max_total_chars`` cap, a
   per-document cap, and ``max_blocks``.
3. **Starvation backfill**: if the gated pack is still under ``target_chars``,
   keep adding gated-out candidates in rank order rather than shipping a starved
   pack - each one marked ``backfill`` so the arm is auditable.
4. **Distractor guard** (``max_marginal_blocks``) caps how many blocks may come
   from the marginal band, so backfill cannot flood the pack with near-misses.
5. **Parent expansion**: a selected ``article_part`` is swapped for its parent
   ``Điều`` when the parent fits the remaining budget - a generator that must
   quote statute wants the whole article, not a fragment.
6. **Nested-span suppression**: never a parent and its own child in one pack.

Defaults reproduce the frozen champion: ``score_floor`` and ``relative_margin``
are ``None``, ``target_chars`` equals ``max_total_chars``, and ordering is
best-first. Nothing changes until an arm is switched on deliberately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ADAPTIVE_PACK_SCHEMA_VERSION = "sedar-adaptive-pack-v1"

SelectionSource = Literal["gated", "backfill", "min_blocks", "parent_expansion"]
OrderPolicy = Literal["best_first", "best_last"]
StopReason = Literal[
    "budget_exhausted",
    "content_exhausted",
    "max_blocks",
    "candidates_exhausted",
    "no_candidates",
]

__all__ = [
    "ADAPTIVE_PACK_SCHEMA_VERSION",
    "AdaptivePackPolicy",
    "AdaptivePackSelection",
    "OrderPolicy",
    "PackCandidate",
    "PackUnit",
    "SelectedBlock",
    "SelectionSource",
    "StopReason",
    "select_adaptive_pack",
]


class PackUnit(Protocol):
    """The unit metadata this module needs.

    Duck-typed on purpose: corpus v4's ``RetrievalUnit`` satisfies it directly,
    and a v3 ``CanonicalPassage`` adapter satisfies it with
    ``parent_unit_id=None`` and ``packable=True``, which makes parent expansion
    and nested suppression no-ops there rather than errors.
    """

    unit_id: str
    document_id: str
    parent_unit_id: str | None
    level: str
    role: str
    char_count: int
    packable: bool


@dataclass(frozen=True, slots=True)
class PackCandidate:
    """One ranked candidate offered to the pack."""

    unit_id: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class SelectedBlock:
    """One candidate that made it into the pack, and why."""

    unit_id: str
    document_id: str
    rank: int
    score: float
    chars: int
    source: SelectionSource
    replaced_unit_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "score": self.score,
            "chars": self.chars,
            "source": self.source,
            "replaced_unit_id": self.replaced_unit_id,
        }


@dataclass(frozen=True, slots=True)
class AdaptivePackPolicy:
    """Budget and gating policy for one pack.

    ``target_chars`` is a *soft floor* the selector fills toward;
    ``max_total_chars`` is the hard cap it must not exceed. Setting them equal
    reproduces the champion's single-cap behaviour. Setting ``target_chars``
    below the cap creates headroom the starvation backfill can use without ever
    growing the pack past the cap.
    """

    min_blocks: int = 1
    max_blocks: int = 6
    target_chars: int = 6000
    max_total_chars: int = 6000
    max_per_document: int = 3
    #: Absolute relevance floor on the reranker score. None disables gating.
    score_floor: float | None = None
    #: Keep candidates within this much of the top score. None disables it.
    relative_margin: float | None = None
    #: Cap on blocks drawn from below the relative margin (the distractor band).
    max_marginal_blocks: int | None = None
    starvation_backfill: bool = True
    #: A pack filling less than this share of ``target_chars`` counts as
    #: starved. It needs a threshold rather than exact equality: a pack almost
    #: never lands exactly on the target, and the measured symptom was specific
    #: - "a quarter of all cases fill less than half the character budget".
    starvation_ratio: float = 0.75
    expand_to_parent: bool = True
    suppress_nested: bool = True
    respect_packable: bool = True
    order: OrderPolicy = "best_first"
    #: A block shorter than this is a fragment; it is skipped while anything
    #: longer is still available. 0 disables the check.
    min_block_chars: int = 0

    def __post_init__(self) -> None:
        if self.min_blocks <= 0:
            raise ValueError("min_blocks must be positive")
        if self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be >= min_blocks")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if self.target_chars > self.max_total_chars:
            raise ValueError("target_chars must not exceed max_total_chars")
        if self.max_per_document <= 0:
            raise ValueError("max_per_document must be positive")
        if self.relative_margin is not None and self.relative_margin < 0.0:
            raise ValueError("relative_margin must be non-negative")
        if self.max_marginal_blocks is not None and self.max_marginal_blocks < 0:
            raise ValueError("max_marginal_blocks must be non-negative")
        if self.min_block_chars < 0:
            raise ValueError("min_block_chars must be non-negative")
        if not 0.0 < self.starvation_ratio <= 1.0:
            raise ValueError("starvation_ratio must be in (0, 1]")
        if self.order not in {"best_first", "best_last"}:
            raise ValueError(f"Unsupported order: {self.order!r}")

    @property
    def gating_enabled(self) -> bool:
        return self.score_floor is not None or self.relative_margin is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_blocks": self.min_blocks,
            "max_blocks": self.max_blocks,
            "target_chars": self.target_chars,
            "max_total_chars": self.max_total_chars,
            "max_per_document": self.max_per_document,
            "score_floor": self.score_floor,
            "relative_margin": self.relative_margin,
            "max_marginal_blocks": self.max_marginal_blocks,
            "starvation_backfill": self.starvation_backfill,
            "starvation_ratio": self.starvation_ratio,
            "expand_to_parent": self.expand_to_parent,
            "suppress_nested": self.suppress_nested,
            "respect_packable": self.respect_packable,
            "order": self.order,
            "min_block_chars": self.min_block_chars,
            "schema_version": ADAPTIVE_PACK_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class AdaptivePackSelection:
    """What the selector chose, and the diagnostics the report needs."""

    blocks: tuple[SelectedBlock, ...]
    total_chars: int
    stop_reason: StopReason
    starved: bool
    fill_ratio: float
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    marginal_blocks: int = 0
    backfilled_blocks: int = 0
    expanded_blocks: int = 0
    suppressed_nested: int = 0
    top_score: float | None = None
    score_span: float | None = None

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(block.unit_id for block in self.blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocks": [block.as_dict() for block in self.blocks],
            "block_count": len(self.blocks),
            "total_chars": self.total_chars,
            "stop_reason": self.stop_reason,
            "starved": self.starved,
            "fill_ratio": round(self.fill_ratio, 4),
            "dropped_reasons": dict(self.dropped_reasons),
            "marginal_blocks": self.marginal_blocks,
            "backfilled_blocks": self.backfilled_blocks,
            "expanded_blocks": self.expanded_blocks,
            "suppressed_nested": self.suppressed_nested,
            "top_score": self.top_score,
            "score_span": self.score_span,
        }


def _chars(unit: PackUnit) -> int:
    return int(unit.char_count or 0)


def _family(unit_id: str, units: Mapping[str, PackUnit]) -> set[str]:
    """The unit plus its parent plus every sibling child of that parent."""

    unit = units.get(unit_id)
    if unit is None:
        return {unit_id}
    family = {unit_id}
    parent_id = unit.parent_unit_id
    if parent_id:
        family.add(parent_id)
    root = parent_id or unit_id
    for other_id, other in units.items():
        if other.parent_unit_id == root or other_id == root:
            family.add(other_id)
    return family


def select_adaptive_pack(
    candidates: Sequence[PackCandidate],
    units: Mapping[str, PackUnit],
    policy: AdaptivePackPolicy,
) -> AdaptivePackSelection:
    """Choose a variable-size pack under a character target and a relevance gate."""

    if not candidates:
        return AdaptivePackSelection(
            blocks=(),
            total_chars=0,
            stop_reason="no_candidates",
            starved=True,
            fill_ratio=0.0,
        )

    known = [item for item in candidates if item.unit_id in units]
    if not known:
        raise KeyError("No candidate unit is present in the corpus view")

    finite = [item.score for item in known if item.score != float("-inf")]
    top_score = max(finite) if finite else None
    floor: float | None = policy.score_floor
    margin_floor: float | None = None
    if policy.relative_margin is not None and top_score is not None:
        margin_floor = top_score - policy.relative_margin
        floor = margin_floor if floor is None else max(floor, margin_floor)

    dropped: dict[str, int] = {}

    def note(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    blocks: list[SelectedBlock] = []
    per_document: dict[str, int] = {}
    chosen_ids: set[str] = set()
    blocked_families: set[str] = set()
    total_chars = 0
    marginal = 0
    backfilled = 0
    expanded = 0
    suppressed = 0
    stop_reason: StopReason = "candidates_exhausted"
    gate_rejected: list[PackCandidate] = []

    def try_add(
        candidate: PackCandidate,
        *,
        source: SelectionSource,
        allow_short: bool,
    ) -> bool:
        nonlocal total_chars, marginal, backfilled, expanded, suppressed
        unit = units[candidate.unit_id]
        if candidate.unit_id in chosen_ids:
            return False
        if policy.suppress_nested and candidate.unit_id in blocked_families:
            note("nested_span")
            suppressed += 1
            return False
        if policy.respect_packable and not unit.packable:
            note("not_packable")
            return False
        if per_document.get(unit.document_id, 0) >= policy.max_per_document:
            note("max_per_document")
            return False

        target_unit = unit
        replaced: str | None = None
        remaining = policy.max_total_chars - total_chars
        if (
            policy.expand_to_parent
            and unit.parent_unit_id
            and unit.parent_unit_id in units
            and unit.parent_unit_id not in chosen_ids
        ):
            parent = units[unit.parent_unit_id]
            parent_chars = _chars(parent)
            fits = parent_chars <= remaining or not blocks
            if parent.packable and fits:
                target_unit = parent
                replaced = unit.unit_id
            else:
                note("parent_did_not_fit")

        size = _chars(target_unit)
        if (
            policy.min_block_chars
            and size < policy.min_block_chars
            and not allow_short
        ):
            note("min_block_chars")
            return False
        if total_chars + size > policy.max_total_chars and blocks:
            note("max_total_chars")
            return False

        blocks.append(
            SelectedBlock(
                unit_id=target_unit.unit_id,
                document_id=target_unit.document_id,
                rank=candidate.rank,
                score=candidate.score,
                chars=size,
                source="parent_expansion" if replaced else source,
                replaced_unit_id=replaced,
            )
        )
        chosen_ids.add(target_unit.unit_id)
        if replaced:
            chosen_ids.add(replaced)
            expanded += 1
        if policy.suppress_nested:
            blocked_families.update(_family(target_unit.unit_id, units))
        per_document[target_unit.document_id] = (
            per_document.get(target_unit.document_id, 0) + 1
        )
        total_chars += size
        if source == "backfill":
            backfilled += 1
        if margin_floor is not None and candidate.score < margin_floor:
            marginal += 1
        return True

    # --- pass 1: gated fill toward the character target -------------------
    for candidate in known:
        if len(blocks) >= policy.max_blocks:
            stop_reason = "max_blocks"
            break
        if total_chars >= policy.target_chars and len(blocks) >= policy.min_blocks:
            stop_reason = "budget_exhausted"
            break
        below_floor = floor is not None and candidate.score < floor
        if below_floor:
            note("score_floor")
            gate_rejected.append(candidate)
            continue
        if (
            policy.max_marginal_blocks is not None
            and margin_floor is not None
            and candidate.score < margin_floor
            and marginal >= policy.max_marginal_blocks
        ):
            note("max_marginal_blocks")
            gate_rejected.append(candidate)
            continue
        try_add(candidate, source="gated", allow_short=False)

    # --- pass 2: starvation backfill --------------------------------------
    # A starved pack is the measured bottleneck, so an under-target pack pulls
    # from the gated-out candidates rather than shipping short. Every block
    # added here is labelled, so the arm can be turned off and measured.
    if (
        policy.starvation_backfill
        and total_chars < policy.target_chars
        and len(blocks) < policy.max_blocks
    ):
        for candidate in gate_rejected:
            if len(blocks) >= policy.max_blocks:
                stop_reason = "max_blocks"
                break
            if total_chars >= policy.target_chars:
                stop_reason = "budget_exhausted"
                break
            if (
                policy.max_marginal_blocks is not None
                and margin_floor is not None
                and candidate.score < margin_floor
                and marginal >= policy.max_marginal_blocks
            ):
                continue
            try_add(candidate, source="backfill", allow_short=False)

    # --- pass 3: min_blocks floor ----------------------------------------
    # An empty or one-block pack because every candidate was filtered is worse
    # than an imperfect pack, and a silently empty one is worst of all.
    if len(blocks) < policy.min_blocks:
        for candidate in known:
            if len(blocks) >= policy.min_blocks:
                break
            try_add(candidate, source="min_blocks", allow_short=True)

    if not blocks:
        head = known[0]
        unit = units[head.unit_id]
        blocks.append(
            SelectedBlock(
                unit_id=head.unit_id,
                document_id=unit.document_id,
                rank=head.rank,
                score=head.score,
                chars=_chars(unit),
                source="min_blocks",
            )
        )
        total_chars = _chars(unit)

    fill_ratio = total_chars / policy.target_chars if policy.target_chars else 0.0
    starved = fill_ratio < policy.starvation_ratio
    if starved and stop_reason == "candidates_exhausted":
        # The budget did not bind; the retriever ran out of usable evidence.
        # This is the distinction the champion's config could not express, and
        # the reason 65 cases were misattributed to the generation cap. A pack
        # that used every candidate and still came out nearly full keeps the
        # benign ``candidates_exhausted``.
        stop_reason = "content_exhausted"

    if policy.order == "best_last":
        # Highest-scoring block adjacent to the question. "The Power of Noise"
        # (SIGIR 2024) measures accuracy highest when the gold document sits
        # next to the query and lowest when farthest; the champion's prompt puts
        # the question after the evidence, so best-last is the arm that tests it.
        blocks.reverse()

    scores = [block.score for block in blocks if block.score != float("-inf")]
    return AdaptivePackSelection(
        blocks=tuple(blocks),
        total_chars=total_chars,
        stop_reason=stop_reason,
        starved=starved,
        fill_ratio=fill_ratio,
        dropped_reasons=dropped,
        marginal_blocks=marginal,
        backfilled_blocks=backfilled,
        expanded_blocks=expanded,
        suppressed_nested=suppressed,
        top_score=top_score,
        score_span=(max(scores) - min(scores)) if len(scores) > 1 else 0.0,
    )
