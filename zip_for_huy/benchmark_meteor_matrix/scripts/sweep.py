#!/usr/bin/env python
"""Run several benchmark configs and compare them with paired bootstrap.

Speeds up the C+B loop: point it at the frozen baseline config plus one or more
variant configs (e.g. different reranker depth, prompt, max_new_tokens). It runs
each, tabulates summary metrics, and reports the paired METEOR delta of every
variant vs the baseline with a 95% bootstrap CI over the shared question ids.

A variant is a real improvement only if its CI lower bound is above zero.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/sweep.py \
        --baseline config/train_dev200.yaml \
        --variants config/variant_a.yaml config/variant_b.yaml \
        --strategy longest_grounded --iters 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.runner import run_from_yaml  # noqa: E402


def _load_strategy_scores(run_dir: Path, strategy: str, metric: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with (run_dir / "per_case.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sel = row["selections"].get(strategy)
            if sel and sel.get(metric) is not None:
                scores[str(row["id"])] = float(sel[metric])
    return scores


def _paired_bootstrap(
    a: np.ndarray, b: np.ndarray, iters: int, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    diff = a - b
    n = len(diff)
    boot = diff[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return float(diff.mean()), float(lo), float(hi), min(p, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline config yaml")
    parser.add_argument("--variants", nargs="*", default=[], help="variant config yamls")
    parser.add_argument("--strategy", default="longest_grounded")
    parser.add_argument("--metric", default="meteor", choices=["meteor", "rouge_l"])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = [args.baseline, *args.variants]
    run_dirs: dict[str, Path] = {}
    summaries: dict[str, dict] = {}
    for cfg in configs:
        out_dir = run_from_yaml(cfg)
        run_dirs[cfg] = out_dir
        summaries[cfg] = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    print(f"{'config':<45}{'meteor':>8}{'rouge_l':>9}{'oracle':>8}{'refuse':>8}{'len':>7}")
    for cfg in configs:
        s = summaries[cfg]
        strat = s["strategies"].get(args.strategy, {})
        print(
            f"{Path(cfg).name:<45}{strat.get('meteor', float('nan')):>8.4f}"
            f"{strat.get('rouge_l', float('nan')):>9.4f}{s['oracle_meteor']:>8.4f}"
            f"{strat.get('refusal_rate', float('nan')):>8.3f}{strat.get('pred_mean_tokens', float('nan')):>7.1f}"
        )

    if args.variants:
        rng = np.random.default_rng(args.seed)
        base_scores = _load_strategy_scores(run_dirs[args.baseline], args.strategy, args.metric)
        print(f"\npaired {args.metric} delta vs baseline ({Path(args.baseline).name}), strategy={args.strategy}")
        print(f"{'variant':<45}{'d':>9}{'ci95':>22}{'p':>8}  sig")
        for cfg in args.variants:
            var_scores = _load_strategy_scores(run_dirs[cfg], args.strategy, args.metric)
            shared = sorted(set(base_scores) & set(var_scores))
            if not shared:
                print(f"{Path(cfg).name:<45}{'n/a':>9}{'no shared ids':>22}")
                continue
            a = np.array([var_scores[i] for i in shared])
            b = np.array([base_scores[i] for i in shared])
            d, lo, hi, p = _paired_bootstrap(a, b, args.iters, rng)
            sig = "*" if lo > 0 else ("-" if hi < 0 else "ns")
            print(f"{Path(cfg).name:<45}{d:>+9.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}{p:>8.3f}  {sig}")
        print("\nsig: '*' improvement (CI>0), '-' regression (CI<0), 'ns' indistinguishable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
