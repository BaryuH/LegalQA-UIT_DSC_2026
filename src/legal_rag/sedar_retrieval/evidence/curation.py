"""Evidence curation + adaptive budget (TASK 19 local deterministic)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Complexity = Literal["simple", "implicit", "multi_condition", "multi_hop"]

DEFAULT_BUDGETS: dict[Complexity, dict[str, int]] = {
    "simple": {"max_blocks": 4, "max_tokens": 2500},
    "implicit": {"max_blocks": 6, "max_tokens": 3500},
    "multi_condition": {"max_blocks": 8, "max_tokens": 4500},
    "multi_hop": {"max_blocks": 10, "max_tokens": 5500},
}


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    passage_id: str
    article_id: str | None
    citation_label: str
    raw_text: str
    score: float
    reason: str = "primary_rule"


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    article_id: str | None
    citation_label: str
    raw_text: str
    source_passage_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class EvidencePack:
    query_id: str
    sufficient: bool
    blocks: tuple[EvidenceBlock, ...]
    token_count: int
    complexity: Complexity


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def curate_evidence(
    *,
    query_id: str,
    candidates: Sequence[CandidateEvidence],
    complexity: Complexity = "simple",
    budgets: dict[Complexity, dict[str, int]] | None = None,
    force_keep_ids: Sequence[str] = (),
) -> EvidencePack:
    """Rank, dedup by article/text, and apply adaptive block/token budgets."""

    cfg = (budgets or DEFAULT_BUDGETS)[complexity]
    max_blocks = cfg["max_blocks"]
    max_tokens = cfg["max_tokens"]

    ordered = sorted(candidates, key=lambda c: (-c.score, c.passage_id))
    selected: list[EvidenceBlock] = []
    seen_text: set[str] = set()
    seen_articles: dict[str, int] = {}
    token_count = 0
    force = set(force_keep_ids)

    for cand in ordered:
        norm = " ".join(cand.raw_text.split())
        if norm in seen_text:
            continue
        if "[SHORT SUMMARY]" in cand.raw_text or "[DOCUMENT CONTEXT]" in cand.raw_text:
            raise ValueError("Synthetic/retrieval context leaked into evidence")
        article_key = cand.article_id or cand.passage_id
        # Keep at most 2 blocks per article unless forced.
        if seen_articles.get(article_key, 0) >= 2 and cand.passage_id not in force:
            continue
        block_tokens = _approx_tokens(cand.raw_text)
        if (
            selected
            and token_count + block_tokens > max_tokens
            and cand.passage_id not in force
        ):
            continue
        if len(selected) >= max_blocks and cand.passage_id not in force:
            break
        selected.append(
            EvidenceBlock(
                article_id=cand.article_id,
                citation_label=cand.citation_label,
                raw_text=cand.raw_text,
                source_passage_ids=(cand.passage_id,),
                reason=cand.reason,
            )
        )
        seen_text.add(norm)
        seen_articles[article_key] = seen_articles.get(article_key, 0) + 1
        token_count += block_tokens

    return EvidencePack(
        query_id=query_id,
        sufficient=True,
        blocks=tuple(selected),
        token_count=token_count,
        complexity=complexity,
    )
