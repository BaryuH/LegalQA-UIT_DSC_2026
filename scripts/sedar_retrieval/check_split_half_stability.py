"""Ask whether a champion's advantage survives being measured on half the data.

About twenty configurations were compared on the same 460 questions and the
winner was taken as the maximum. That is selection bias: the reported margin
is optimistic even when every individual comparison was sound. A recorded
data point puts the size of it in view - the public pair was METEOR 0.4894
against 0.5010 measured locally at the time.

This does not correct the bias. It asks a narrower, answerable question: is
the advantage a property of the data or of the particular set of cases? The
cases are split at random many times over, and the candidate's margin is
measured on each half independently. An advantage that is real appears in
both halves nearly every time. One fitted to noise flips.

Reports only counts and numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from statistics import mean, median

_PAIRED = Path(__file__).with_name("compare_metrics_paired.py")
_spec = importlib.util.spec_from_file_location("_paired", _PAIRED)
assert _spec and _spec.loader
_paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paired)


def split_half_stability(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    *,
    metric: str = "meteor",
    splits: int = 200,
    seed: int = 20260906,
) -> dict:
    shared = sorted(set(baseline) & set(candidate))
    deltas = {
        case_id: candidate[case_id][metric] - baseline[case_id][metric]
        for case_id in shared
        if baseline[case_id].get(metric) is not None
        and candidate[case_id].get(metric) is not None
    }
    ids = sorted(deltas)
    if len(ids) < 4:
        raise SystemExit(f"Not enough shared cases for {metric}: {len(ids)}")

    rng = random.Random(seed)
    both_positive = 0
    disagreements = 0
    halves: list[tuple[float, float]] = []
    for _ in range(splits):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = len(shuffled) // 2
        a = mean(deltas[i] for i in shuffled[:cut])
        b = mean(deltas[i] for i in shuffled[cut:])
        halves.append((a, b))
        if a > 0 and b > 0:
            both_positive += 1
        if (a > 0) != (b > 0):
            disagreements += 1

    spread = [abs(a - b) for a, b in halves]
    return {
        "metric": metric,
        "n": len(ids),
        "splits": splits,
        "full_delta": mean(deltas.values()),
        "both_halves_positive": both_positive / splits,
        "halves_disagree_on_sign": disagreements / splits,
        "median_gap_between_halves": median(spread),
        "worst_half_seen": min(min(a, b) for a, b in halves),
        "best_half_seen": max(max(a, b) for a, b in halves),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--metrics", nargs="+", default=["meteor", "rouge_l"])
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args(argv)

    base = _paired.load_per_case(args.baseline)
    cand = _paired.load_per_case(args.candidate)
    for metric in args.metrics:
        r = split_half_stability(
            base, cand, metric=metric, splits=args.splits, seed=args.seed
        )
        print(f"\n=== {metric} (n={r['n']}, {r['splits']} lan chia doi) ===")
        print(f"  delta tren toan bo          : {r['full_delta']:+.4f}")
        print(f"  ca hai nua deu duong        : {r['both_halves_positive']:.1%}")
        print(f"  hai nua NGUOC dau           : {r['halves_disagree_on_sign']:.1%}")
        print(f"  chenh lech trung vi hai nua : {r['median_gap_between_halves']:.4f}")
        print(f"  nua te nhat / tot nhat      : "
              f"{r['worst_half_seen']:+.4f} / {r['best_half_seen']:+.4f}")
        if r["both_halves_positive"] >= 0.95:
            print("  -> ben: loi the xuat hien o gan nhu moi cach chia")
        elif r["halves_disagree_on_sign"] >= 0.20:
            print("  -> MONG MANH: dau hieu bam nhieu, doc lai quyet dinh nay")
        else:
            print("  -> trung binh: that nhung khong day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
