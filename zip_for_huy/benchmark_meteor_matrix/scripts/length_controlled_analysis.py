#!/usr/bin/env python
"""Length-controlled analysis of a benchmark run.

Problem this answers: on the grounded dev slice the strategy ranking is rank-1.0
correlated with the mean length each strategy picks (candidates ~3x shorter than
gold, and METEOR is recall-weighted). So "longest wins" and "mbr is mid" may be
pure length effects, not selection quality.

This isolates *content* selection from *length* by:
  1. joining per_case.jsonl (per-candidate gold METEOR + each strategy's picked
     index) with candidates.jsonl (candidate texts -> token lengths);
  2. removing per-case difficulty (case-mean demeaning of both METEOR and length);
  3. regressing demeaned METEOR on demeaned length (pooled OLS) to get the
     length slope, and taking the residual = the part of METEOR NOT explained by
     length;
  4. reporting, per strategy, the mean length-residual of its picks. A strategy
     that genuinely selects better *content* has a positive residual; a strategy
     that only exploits length (e.g. ``longest``) sits near zero or negative.

It also reports, per strategy, the within-case METEOR-rank minus length-rank of
its pick: positive means it beats what its length alone would buy.

Read-only over artifacts; touches no shared benchmark code.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/length_controlled_analysis.py \
        --run-dir benchmark_meteor_matrix/outputs/<run_name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import tokenize  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rank01(values: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1]; ties share the average position."""

    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float)
    n = len(values)
    return order / (n - 1) if n > 1 else np.zeros(n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidates", default=None, help="candidates.jsonl (default: run-dir/candidates.jsonl)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    per_case = _read_jsonl(run_dir / "per_case.jsonl")
    cand_path = Path(args.candidates) if args.candidates else run_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise SystemExit(f"candidates file not found: {cand_path}")
    cand_by_id = {str(r["case_id"]): list(r["candidates"]) for r in _read_jsonl(cand_path)}

    # Pool per-candidate (length, meteor) with case ids for demeaning.
    pool_len: list[float] = []
    pool_met: list[float] = []
    pool_case: list[int] = []
    # Per strategy: list of (case_index, picked_candidate_index)
    picks: dict[str, list[tuple[int, int]]] = {}
    # Per candidate cache of lengths, indexed by case position.
    case_lengths: list[np.ndarray] = []
    case_meteors: list[np.ndarray] = []

    for ci, row in enumerate(per_case):
        cid = str(row["id"])
        cands = cand_by_id.get(cid)
        met = row.get("gold_meteor_per_candidate")
        if not cands or not met or len(cands) != len(met):
            case_lengths.append(np.zeros(0))
            case_meteors.append(np.zeros(0))
            continue
        lens = np.array([len(tokenize(c)) for c in cands], dtype=np.float64)
        mets = np.array(met, dtype=np.float64)
        case_lengths.append(lens)
        case_meteors.append(mets)
        for length, meteor in zip(lens, mets):
            pool_len.append(length)
            pool_met.append(meteor)
            pool_case.append(ci)
        for name, sel in row["selections"].items():
            idx = sel.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(cands):
                picks.setdefault(name, []).append((ci, idx))

    if not pool_len:
        raise SystemExit("no joinable cases (candidates.jsonl / per_case.jsonl mismatch)")

    plen = np.array(pool_len)
    pmet = np.array(pool_met)
    pcase = np.array(pool_case)

    # Case-mean demeaning removes per-question difficulty.
    dlen = plen.copy()
    dmet = pmet.copy()
    for ci in np.unique(pcase):
        m = pcase == ci
        dlen[m] -= plen[m].mean()
        dmet[m] -= pmet[m].mean()

    # Pooled OLS slope of demeaned meteor on demeaned length.
    denom = float((dlen * dlen).sum())
    slope = float((dlen * dmet).sum() / denom) if denom > 1e-12 else 0.0

    raw_corr = float(np.corrcoef(plen, pmet)[0, 1])
    within_corr = float(np.corrcoef(dlen, dmet)[0, 1]) if dlen.std() > 0 else float("nan")
    print(f"run: {run_dir.name}   candidates joined: {len(plen)} across {len(np.unique(pcase))} cases")
    print(f"corr(length, meteor):  raw={raw_corr:+.3f}   within-case={within_corr:+.3f}")
    print(f"length slope (within-case OLS): {slope:+.5f} meteor per token")
    print("A strategy adds real content only if length_residual > 0.\n")

    # Per candidate: within-case meteor rank and length rank in [0,1].
    met_rank = {}
    len_rank = {}
    for ci, (lens, mets) in enumerate(zip(case_lengths, case_meteors)):
        if len(lens):
            met_rank[ci] = _rank01(mets)
            len_rank[ci] = _rank01(lens)

    rows = []
    for name, sel_list in picks.items():
        residuals = []
        pick_len = []
        pick_met = []
        rank_gap = []
        for ci, idx in sel_list:
            length = case_lengths[ci][idx]
            meteor = case_meteors[ci][idx]
            case_mean_len = case_lengths[ci].mean()
            case_mean_met = case_meteors[ci].mean()
            residuals.append((meteor - case_mean_met) - slope * (length - case_mean_len))
            pick_len.append(length)
            pick_met.append(meteor)
            rank_gap.append(met_rank[ci][idx] - len_rank[ci][idx])
        rows.append(
            {
                "name": name,
                "meteor": float(np.mean(pick_met)),
                "pick_len": float(np.mean(pick_len)),
                "length_residual": float(np.mean(residuals)),
                "rank_gap": float(np.mean(rank_gap)),
            }
        )

    rows.sort(key=lambda r: r["length_residual"], reverse=True)
    print(f"{'strategy':<20}{'meteor':>8}{'pick_len':>9}{'len_resid':>11}{'rank_gap':>10}")
    for r in rows:
        print(
            f"{r['name']:<20}{r['meteor']:>8.4f}{r['pick_len']:>9.1f}"
            f"{r['length_residual']:>+11.4f}{r['rank_gap']:>+10.3f}"
        )
    print("\nlen_resid: METEOR above what the pick's length predicts (content signal).")
    print("rank_gap:  within-case METEOR-rank minus length-rank of the pick (>0 good).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
