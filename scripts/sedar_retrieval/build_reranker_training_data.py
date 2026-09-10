#!/usr/bin/env python3
"""Build (query, positive, semi-hard negatives) pairs for reranker fine-tuning.

**Read this before changing the negative policy.** On the closest published
setup to ours - SoICT Hackathon 2024, a 261,446-document Vietnamese legal corpus
(arXiv:2507.14619, also ICCCI 2025) - the choice of negatives dominated every
other training decision, and the intuitive choice was the worst one:

    baseline (no fine-tune)   MRR@10 0.5584
    hard negatives,  n=2      MRR@10 0.2689   <- halved
    hard negatives,  n=5      MRR@10 0.4796
    hard negatives,  n=10     MRR@10 0.6751
    easy negatives,  n=10     MRR@10 0.5940
    semi-hard,       n=2      MRR@10 0.7681
    semi-hard,       n=5      MRR@10 0.7821
    semi-hard,       n=10     MRR@10 0.7911   <- best, +0.233 over baseline

The mechanism is measured too: **50.87% of their "hard" negatives sat at cosine
>= 0.9 with the positive** (mean 0.6806), i.e. most of them were false
negatives. Semi-hard negatives had 79.44% below 0.5 (mean 0.2072). In a legal
corpus this is not an artefact - neighbouring articles of the same Điều
genuinely co-answer a question, and our own gold answers cite 1.65 distinct Điều
on average with 38% citing two or more. Training a reranker to push those away
teaches it to reject correct evidence.

So this builder mines a *band*, not a top-k. A candidate is accepted as a
semi-hard negative only when its per-query normalised retriever score sits
strictly between ``--easy-below`` and ``--false-negative-above``. Everything
above the upper bound is treated as a suspected false negative and excluded;
everything below the lower bound is too easy to teach anything. The band, and
the share of candidates each rule removed, are written to the audit so the
distribution can be inspected before any GPU time is spent - and the run fails
closed if the accepted band looks like the hard-negative failure case.

Silver retrieval labels are read here because this is an approved training task;
the questions are loaded through ``load_inference_questions``, so no gold answer
text is materialised at all. Nothing this script writes may enter a retrieval
query, an index, an inference prompt or a submission artifact.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import load_rerank_units

RERANKER_DATA_SCHEMA_VERSION = "sedar-reranker-train-data-v1"


def _require_file(path: Path, flag: str) -> Path:
    if str(path) in {"", "."}:
        raise SystemExit(f"{flag} resolved to an empty path; export the variable")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist or is not a file: {path}")
    return path


def _load_labels(path: Path) -> dict[str, set[str]]:
    """query_id -> set of gold unit ids (silver labels are fine here)."""

    labels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or row.get("id") or "").strip()
            if not query_id:
                raise SystemExit(f"Label row without query_id in {path}")
            for key in (
                "gold_unit_ids",
                "gold_passage_ids",
                "passage_ids",
                "relevant_ids",
                "labels",
            ):
                value = row.get(key)
                if isinstance(value, list):
                    labels[query_id].update(str(item) for item in value)
            single = row.get("passage_id") or row.get("unit_id")
            if single:
                labels[query_id].add(str(single))
    resolved = {key: value for key, value in labels.items() if value}
    if not resolved:
        raise SystemExit(f"No usable labels found in {path}")
    return resolved


def _load_candidates(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    """query_id -> [(unit_id, raw retriever score, rank)] in ranked order."""

    candidates: dict[str, list[tuple[str, float, int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                raise SystemExit(f"Candidate row without query_id in {path}")
            rows: list[tuple[str, float, int]] = []
            payload = row.get("candidates")
            if isinstance(payload, list) and payload:
                for rank, item in enumerate(payload, start=1):
                    unit_id = str(item.get("passage_id") or item.get("unit_id") or "")
                    if not unit_id:
                        continue
                    score = item.get("fusion_score")
                    if score is None:
                        score = item.get("rrf_score")
                    if score is None:
                        score = item.get("dense_score")
                    if score is None:
                        score = item.get("bm25_score")
                    if score is None:
                        score = 1.0 / (60 + rank)
                    rows.append((unit_id, float(score), rank))
            else:
                for rank, unit_id in enumerate(row.get("ranked_ids") or [], start=1):
                    rows.append((str(unit_id), 1.0 / (60 + rank), rank))
            if rows:
                candidates[query_id] = rows
    if not candidates:
        raise SystemExit(f"No candidate rows found in {path}")
    return candidates


def _normalise(scores: list[float]) -> list[float]:
    """Per-query min-max so the band is comparable across queries."""

    if not scores:
        return []
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 0.0:
        return [1.0] * len(scores)
    return [(value - low) / span for value in scores]


def _normalise_ranks(rows: list[tuple[str, float, int]]) -> list[float]:
    """Map rank 1 to 1 and the last observed rank to 0."""

    if not rows:
        return []
    last_rank = max(rank for _, _, rank in rows)
    span = last_rank - 1
    if span <= 0:
        return [1.0] * len(rows)
    return [(last_rank - rank) / span for _, _, rank in rows]


def _children_by_parent(units: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Index containment once instead of scanning the corpus per positive."""

    children: dict[str, list[str]] = defaultdict(list)
    for unit_id, unit in units.items():
        parent_id = unit.parent_unit_id
        if parent_id:
            children[parent_id].append(unit_id)
    return {
        parent_id: tuple(sorted(unit_ids)) for parent_id, unit_ids in children.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--negatives",
        type=int,
        default=10,
        help=(
            "Negatives per positive. 10 is the measured optimum for semi-hard "
            "mining on a Vietnamese legal corpus; do not lower it to 2."
        ),
    )
    parser.add_argument(
        "--false-negative-above",
        type=float,
        default=0.75,
        help=(
            "Candidates whose normalised score exceeds this are treated as "
            "suspected false negatives and excluded, not used as negatives."
        ),
    )
    parser.add_argument(
        "--easy-below",
        type=float,
        default=0.15,
        help="Candidates below this are too easy to teach anything.",
    )
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument(
        "--score-mode",
        choices=("raw", "rank"),
        default="raw",
        help="Normalize fused scores or observed candidate ranks per query.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-band-share",
        type=float,
        default=0.20,
        help=(
            "Fail closed when fewer than this share of examined candidates land "
            "inside the semi-hard band: the band is then misconfigured for this "
            "score distribution and the run would train on the wrong negatives."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.easy_below < args.false_negative_above <= 1.0:
        raise SystemExit("require 0 <= --easy-below < --false-negative-above <= 1")
    if args.negatives <= 0:
        raise SystemExit("--negatives must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    for path, flag in (
        (args.labels, "--labels"),
        (args.candidates, "--candidates"),
        (args.units, "--units"),
        (args.questions, "--questions"),
    ):
        _require_file(path, flag)

    labels = _load_labels(args.labels)
    candidates = _load_candidates(args.candidates)
    units = load_rerank_units(args.units)
    children_by_parent = _children_by_parent(units)
    questions = {
        item.id: item.question
        for item in load_inference_questions(args.questions, split=args.split)
    }

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.jsonl"

    counters = {
        "queries_seen": 0,
        "queries_written": 0,
        "queries_without_positive_in_view": 0,
        "queries_without_enough_negatives": 0,
        "candidates_examined": 0,
        "excluded_gold": 0,
        "excluded_same_article": 0,
        "excluded_suspected_false_negative": 0,
        "excluded_too_easy": 0,
        "excluded_beyond_max_rank": 0,
        "accepted_negatives": 0,
        "positive_pairs": 0,
    }
    band_scores: list[float] = []
    excluded_scores: list[float] = []

    with pairs_path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(labels):
            counters["queries_seen"] += 1
            question = questions.get(query_id)
            rows = candidates.get(query_id)
            if question is None or not rows:
                continue

            gold_ids = {gid for gid in labels[query_id] if gid in units}
            if not gold_ids:
                counters["queries_without_positive_in_view"] += 1
                continue

            # Containment leakage: a gold article's own children (and a gold
            # child's parent) are the same law text at another granularity. They
            # are not negatives.
            forbidden = set(gold_ids)
            for gid in gold_ids:
                gold_unit = units[gid]
                if gold_unit.parent_unit_id:
                    forbidden.add(gold_unit.parent_unit_id)
                forbidden.update(children_by_parent.get(gid, ()))

            normalised = (
                _normalise_ranks(rows)
                if args.score_mode == "rank"
                else _normalise([score for _, score, _ in rows])
            )
            band: list[str] = []
            for (unit_id, _, rank), norm in zip(rows, normalised, strict=True):
                counters["candidates_examined"] += 1
                if rank > args.max_rank:
                    counters["excluded_beyond_max_rank"] += 1
                    continue
                if unit_id not in units:
                    continue
                if unit_id in gold_ids:
                    counters["excluded_gold"] += 1
                    continue
                if unit_id in forbidden:
                    counters["excluded_same_article"] += 1
                    continue
                if norm > args.false_negative_above:
                    counters["excluded_suspected_false_negative"] += 1
                    excluded_scores.append(norm)
                    continue
                if norm < args.easy_below:
                    counters["excluded_too_easy"] += 1
                    continue
                band.append(unit_id)
                band_scores.append(norm)

            if len(band) < args.negatives:
                counters["queries_without_enough_negatives"] += 1
                if not band:
                    continue
            chosen = (
                band
                if len(band) <= args.negatives
                else rng.sample(band, args.negatives)
            )
            counters["accepted_negatives"] += len(chosen)

            for gid in sorted(gold_ids):
                counters["positive_pairs"] += 1
                handle.write(
                    json.dumps(
                        {
                            "schema_version": RERANKER_DATA_SCHEMA_VERSION,
                            "query_id": query_id,
                            "query": question,
                            "positive_id": gid,
                            "positive": units[gid].reader_text,
                            "negative_ids": chosen,
                            "negatives": [units[nid].reader_text for nid in chosen],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            counters["queries_written"] += 1

    examined = max(1, counters["candidates_examined"])
    band_share = counters["accepted_negatives"] / examined
    audit: dict[str, Any] = {
        "schema_version": RERANKER_DATA_SCHEMA_VERSION,
        "policy": {
            "negatives": args.negatives,
            "false_negative_above": args.false_negative_above,
            "easy_below": args.easy_below,
            "max_rank": args.max_rank,
            "score_mode": args.score_mode,
            "seed": args.seed,
        },
        "counters": counters,
        "band_share_of_examined": round(band_share, 4),
        "suspected_false_negative_share": round(
            counters["excluded_suspected_false_negative"] / examined, 4
        ),
        "band_score": {
            "mean": round(statistics.mean(band_scores), 4) if band_scores else 0.0,
            "median": round(statistics.median(band_scores), 4) if band_scores else 0.0,
        },
        "excluded_score": {
            "mean": (
                round(statistics.mean(excluded_scores), 4) if excluded_scores else 0.0
            ),
        },
        "pairs_path": str(pairs_path),
    }

    gate_failures: list[str] = []
    if counters["positive_pairs"] == 0:
        gate_failures.append("no positive pairs were written")
    if band_share < args.min_band_share:
        gate_failures.append(
            f"only {band_share:.3f} of examined candidates fell inside the "
            f"semi-hard band (< --min-band-share={args.min_band_share}); the "
            "band bounds do not match this score distribution"
        )
    if band_scores and statistics.mean(band_scores) > args.false_negative_above:
        gate_failures.append(
            "the accepted band's mean score exceeds the false-negative bound, "
            "which is the hard-negative failure case"
        )
    audit["gate_failures"] = gate_failures
    audit["status"] = "FAIL" if gate_failures else "PASS"

    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if gate_failures:
        raise SystemExit(
            "Refusing to hand these pairs to training; see audit.json gate_failures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
