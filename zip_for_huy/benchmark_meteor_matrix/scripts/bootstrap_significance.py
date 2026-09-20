#!/usr/bin/env python
"""Paired bootstrap significance over a benchmark run's per_case.jsonl.

The selection deltas are small and the dev slice is short, so a raw ranking of
mean METEOR is not enough to crown a winner. This computes, for a chosen
baseline strategy, the paired METEOR difference of every other strategy with a
95% bootstrap CI and a two-sided p-value. A strategy is only a real winner if
its CI lower bound is above zero.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/bootstrap_significance.py \
        --run-dir benchmark_meteor_matrix/outputs/<run_name> \
        --baseline first --metric meteor --iters 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_scores(run_dir: Path, metric: str) -> dict[str, np.ndarray]:
    rows = []
    with (run_dir / "per_case.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    strategies = sorted({name for r in rows for name in r["selections"]})
    scores: dict[str, list[float]] = {s: [] for s in strategies}
    for r in rows:
        for s in strategies:
            sel = r["selections"].get(s)
            scores[s].append(float(sel[metric]) if sel and sel[metric] is not None else 0.0)
    return {s: np.asarray(v, dtype=np.float64) for s, v in scores.items()}


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, iters: int, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """Return (observed_diff, ci_low, ci_high, two_sided_p) for mean(a - b)."""

    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(iters, n))
    boot = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    frac_le0 = float(np.mean(boot <= 0.0))
    frac_ge0 = float(np.mean(boot >= 0.0))
    p = 2.0 * min(frac_le0, frac_ge0)
    return observed, float(ci_low), float(ci_high), min(p, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline", default="first")
    parser.add_argument("--metric", default="meteor", choices=["meteor", "rouge_l"])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores = load_scores(run_dir, args.metric)
    if args.baseline not in scores:
        raise SystemExit(f"baseline {args.baseline!r} not in {sorted(scores)}")
    rng = np.random.default_rng(args.seed)
    base = scores[args.baseline]
    n = len(base)

    ranked = sorted(scores.items(), key=lambda kv: kv[1].mean(), reverse=True)
    print(f"run: {run_dir.name}  metric: {args.metric}  cases: {n}  "
          f"baseline: {args.baseline}  iters: {args.iters}")
    print(f"{'strategy':<20}{'mean':>8}{'d_vs_base':>11}{'ci95':>20}{'p':>9}  sig")
    for name, arr in ranked:
        if name == args.baseline:
            print(f"{name:<20}{arr.mean():>8.4f}{'n/a':>11}{'(baseline)':>20}{'':>9}")
            continue
        d, lo, hi, p = paired_bootstrap(arr, base, args.iters, rng)
        sig = "*" if lo > 0 else ("-" if hi < 0 else "ns")
        print(f"{name:<20}{arr.mean():>8.4f}{d:>+11.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{p:>9.3f}  {sig}")
    print("\nsig: '*' = better than baseline (CI>0), '-' = worse (CI<0), "
          "'ns' = not distinguishable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
