#!/usr/bin/env python3
"""Select adaptive evidence packs and report the distribution (TASK 25).

Reads a ranking (ideally the reranked one from
``run_vietnamese_reranker.py``, which carries cross-encoder scores) plus the
corpus units, applies the adaptive pack policy, and writes one pack per query
along with the report needed to calibrate the policy.

Calibration is a two-pass job, and the first pass must be the control:

    pass 1  --score-floor unset, --target-chars = --max-total-chars
            reproduces the champion's fixed budget and dumps the distribution
    pass 2  set --score-floor from ``score_floor_percentiles`` in that report,
            and --target-chars below the cap so backfill has headroom

The report's two most important fields are ``stop_reasons`` and
``starvation_rate``. ``content_exhausted`` means the retriever ran out of usable
evidence; ``budget_exhausted`` means evidence existed and the cap refused it.
Those call for opposite fixes, and the champion's config could not tell them
apart - which is why 65 UNDER_SPECIFIED cases were misattributed to the
generation cap.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.evidence.adaptive_pack import (
    ADAPTIVE_PACK_SCHEMA_VERSION,
    AdaptivePackPolicy,
    PackCandidate,
    select_adaptive_pack,
)
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import load_rerank_units


def _require_file(path: Path, flag: str) -> Path:
    if str(path) in {"", "."}:
        raise SystemExit(f"{flag} resolved to an empty path; export the variable")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist or is not a file: {path}")
    return path


def _load_scores(path: Path | None) -> dict[str, dict[str, float]]:
    """query_id -> {unit_id: cross-encoder score}, from a scores JSONL."""

    if path is None:
        return {}
    scores: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                continue
            per_unit: dict[str, float] = {}
            for item in row.get("scores") or []:
                unit_id = str(item.get("unit_id") or item.get("passage_id") or "")
                if not unit_id:
                    continue
                value = item.get("ce_score")
                if value is None:
                    value = item.get("score")
                if value is not None:
                    per_unit[unit_id] = float(value)
            if per_unit:
                scores[query_id] = per_unit
    return scores


def _load_ranking(path: Path) -> dict[str, list[tuple[str, float | None]]]:
    """query_id -> [(unit_id, inline score or None)] in ranked order."""

    rankings: dict[str, list[tuple[str, float | None]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                raise SystemExit(f"Ranking row without query_id in {path}")
            inline: dict[str, float] = {}
            for item in row.get("candidates") or row.get("scores") or []:
                unit_id = str(item.get("unit_id") or item.get("passage_id") or "")
                if not unit_id:
                    continue
                for key in ("ce_score", "fusion_score", "rrf_score", "score"):
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        inline[unit_id] = float(value)
                        break
            ranked = row.get("ranked_ids")
            if not isinstance(ranked, list) or not ranked:
                raise SystemExit(f"Query {query_id!r} has no ranked_ids in {path}")
            if query_id in rankings:
                raise SystemExit(f"Duplicate query_id {query_id!r} in {path}")
            rankings[query_id] = [
                (str(unit_id), inline.get(str(unit_id))) for unit_id in ranked
            ]
    if not rankings:
        raise SystemExit(f"No ranking rows found in {path}")
    return rankings


def _percentiles(values: list[float], places: int = 4) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], places)

    return {
        "p05": at(0.05),
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional ce_scores.jsonl from run_vietnamese_reranker.py.",
    )
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidate-window", type=int, default=32)
    # Champion control arm: 6 blocks / 6000 chars / 3 per document.
    parser.add_argument("--min-blocks", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=6)
    parser.add_argument("--target-chars", type=int, default=6000)
    parser.add_argument("--max-total-chars", type=int, default=6000)
    parser.add_argument("--max-per-document", type=int, default=3)
    parser.add_argument("--score-floor", type=float, default=None)
    parser.add_argument("--relative-margin", type=float, default=None)
    parser.add_argument("--max-marginal-blocks", type=int, default=None)
    parser.add_argument("--min-block-chars", type=int, default=0)
    parser.add_argument("--no-starvation-backfill", action="store_true")
    parser.add_argument("--no-parent-expansion", action="store_true")
    parser.add_argument("--no-nested-suppression", action="store_true")
    parser.add_argument("--ignore-packable", action="store_true")
    parser.add_argument("--order", choices=("best_first", "best_last"), default="best_first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.candidate_window <= 0:
        raise SystemExit("--candidate-window must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")
    _require_file(args.ranking, "--ranking")
    _require_file(args.units, "--units")
    if args.scores is not None:
        _require_file(args.scores, "--scores")

    rankings = _load_ranking(args.ranking)
    external_scores = _load_scores(args.scores)
    units = load_rerank_units(args.units)

    policy = AdaptivePackPolicy(
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        target_chars=args.target_chars,
        max_total_chars=args.max_total_chars,
        max_per_document=args.max_per_document,
        score_floor=args.score_floor,
        relative_margin=args.relative_margin,
        max_marginal_blocks=args.max_marginal_blocks,
        starvation_backfill=not args.no_starvation_backfill,
        expand_to_parent=not args.no_parent_expansion,
        suppress_nested=not args.no_nested_suppression,
        respect_packable=not args.ignore_packable,
        order=args.order,
        min_block_chars=args.min_block_chars,
    )

    block_counts: list[int] = []
    char_totals: list[int] = []
    fill_ratios: list[float] = []
    top_scores: list[float] = []
    # The score of the last block that a *fixed* champion-sized pack would have
    # taken. This is the distribution to read a --score-floor off: a floor below
    # it changes nothing, a floor above it starts trimming.
    marginal_scores: list[float] = []
    stop_reasons: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    starved = 0
    backfilled = 0
    expanded = 0
    suppressed = 0
    missing_units: Counter[str] = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for query_id in sorted(rankings):
            per_query = external_scores.get(query_id, {})
            candidates: list[PackCandidate] = []
            for rank, (unit_id, inline) in enumerate(
                rankings[query_id][: args.candidate_window], start=1
            ):
                if unit_id not in units:
                    missing_units[unit_id] += 1
                    continue
                score = per_query.get(unit_id, inline)
                if score is None:
                    # No score anywhere: fall back to a rank-decreasing proxy so
                    # ordering is preserved and gating degrades to a no-op.
                    score = 1.0 / rank
                candidates.append(
                    PackCandidate(unit_id=unit_id, rank=rank, score=float(score))
                )
            if not candidates:
                raise SystemExit(
                    f"Query {query_id!r} has no candidate present in --units; "
                    "the ranking and the corpus view disagree"
                )

            selection = select_adaptive_pack(candidates, units, policy)
            block_counts.append(len(selection.blocks))
            char_totals.append(selection.total_chars)
            fill_ratios.append(selection.fill_ratio)
            if selection.top_score is not None:
                top_scores.append(selection.top_score)
            if len(candidates) >= args.max_blocks:
                marginal_scores.append(candidates[args.max_blocks - 1].score)
            stop_reasons[selection.stop_reason] += 1
            dropped.update(selection.dropped_reasons)
            starved += int(selection.starved)
            backfilled += selection.backfilled_blocks
            expanded += selection.expanded_blocks
            suppressed += selection.suppressed_nested

            handle.write(
                json.dumps(
                    {
                        "schema_version": ADAPTIVE_PACK_SCHEMA_VERSION,
                        "query_id": query_id,
                        "policy": policy.as_dict(),
                        "pack_ids": list(selection.ordered_ids),
                        "pack": selection.as_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    queries = max(1, len(rankings))
    report: dict[str, Any] = {
        "schema_version": ADAPTIVE_PACK_SCHEMA_VERSION,
        "ranking": str(args.ranking),
        "scores": str(args.scores) if args.scores else None,
        "units": str(args.units),
        "candidate_window": args.candidate_window,
        "policy": policy.as_dict(),
        "queries": len(rankings),
        "block_count": {
            "mean": round(statistics.mean(block_counts), 3) if block_counts else 0,
            "median": statistics.median(block_counts) if block_counts else 0,
            "min": min(block_counts) if block_counts else 0,
            "max": max(block_counts) if block_counts else 0,
            "histogram": dict(sorted(Counter(block_counts).items())),
        },
        "pack_chars": {
            "mean": round(statistics.mean(char_totals), 1) if char_totals else 0,
            **_percentiles([float(value) for value in char_totals], places=0),
        },
        "fill_ratio": {
            "mean": round(statistics.mean(fill_ratios), 4) if fill_ratios else 0,
            **_percentiles(fill_ratios),
        },
        # The headline diagnostic: what share of packs ship under target.
        "starvation_rate": round(starved / queries, 4),
        "stop_reasons": dict(stop_reasons),
        "dropped_reasons": dict(dropped),
        "backfilled_blocks_total": backfilled,
        "expanded_blocks_total": expanded,
        "suppressed_nested_total": suppressed,
        "top_score_percentiles": _percentiles(top_scores),
        "score_floor_percentiles": _percentiles(marginal_scores),
        "missing_unit_count": len(missing_units),
        "output": str(args.output),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
