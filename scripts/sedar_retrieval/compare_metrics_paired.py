"""Paired comparison of two scored runs over the cases they share.

Every adoption decision in this project rests on a paired test: two runs
scored on the same cases, compared case by case, with a bootstrap interval
around the mean difference. The tool that did it lived in /tmp and was lost
once. It belongs in the repository.

Paired, not two independent means: the two runs answer the same questions, so
per-case pairing removes question difficulty from the comparison and leaves
far less variance than comparing aggregates.

Reports only IDs and numbers. Never prints an answer or a reference.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

DEFAULT_METRICS = ("meteor", "rouge_l")


def load_per_case(path: Path) -> dict[str, dict]:
    """Read a scorer artifact into {case_id: row}."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_case")
    if not isinstance(rows, list):
        raise SystemExit(f"{path} has no per_case list")
    out: dict[str, dict] = {}
    for row in rows:
        case_id = str(row["id"])
        if case_id in out:
            raise SystemExit(f"Duplicate case id in {path}: {case_id}")
        out[case_id] = row
    return out


def bootstrap_ci(
    deltas: list[float],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""

    if not deltas:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        means.append(mean(deltas[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


def permutation_p(deltas: list[float], *, resamples: int, seed: int) -> float:
    """Two-sided sign-flip permutation p-value for a zero mean difference.

    Sign flipping is the exchangeability that pairing buys: under the null,
    which run a case favours is arbitrary.
    """

    if not deltas:
        return float("nan")
    observed = abs(mean(deltas))
    rng = random.Random(seed)
    hits = 0
    for _ in range(resamples):
        flipped = mean(d if rng.random() < 0.5 else -d for d in deltas)
        if abs(flipped) >= observed:
            hits += 1
    return (hits + 1) / (resamples + 1)


def compare(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    *,
    metrics=DEFAULT_METRICS,
    resamples: int = 10000,
    seed: int = 20260906,
    noise_floor: float = 0.008,
) -> dict:
    shared = sorted(set(baseline) & set(candidate))
    result: dict = {"shared_cases": len(shared), "metrics": {}}
    for metric in metrics:
        pairs = [
            (baseline[i].get(metric), candidate[i].get(metric))
            for i in shared
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        deltas = [b - a for a, b in pairs]
        if not deltas:
            continue
        lo, hi = bootstrap_ci(deltas, resamples=resamples, seed=seed)
        result["metrics"][metric] = {
            "n": len(deltas),
            "baseline": mean(a for a, _ in pairs),
            "candidate": mean(b for _, b in pairs),
            "delta": mean(deltas),
            "ci_low": lo,
            "ci_high": hi,
            "p_value": permutation_p(deltas, resamples=resamples, seed=seed + 1),
            "better": sum(1 for d in deltas if d > 0),
            "worse": sum(1 for d in deltas if d < 0),
            # Judge the interval, not the point estimate. Four outcomes, and
            # the fourth matters: a wide interval spanning the floor in both
            # directions is INCONCLUSIVE, not a regression. Reading such a
            # result as a drop because its lower bound is low would condemn
            # an intervention the data cannot judge.
            "beats_noise_floor": lo > noise_floor,
            "within_noise_floor": lo >= -noise_floor and hi <= noise_floor,
            "is_regression": hi < -noise_floor,
            "verdict": (
                "vuot_noise_floor" if lo > noise_floor
                else "tut_ro_rang" if hi < -noise_floor
                else "trong_noise_floor"
                if (lo >= -noise_floor and hi <= noise_floor)
                else "khong_ket_luan_duoc"
            ),
        }
    return result


def _render(name: str, r: dict) -> None:
    print(f"shared cases: {r['shared_cases']}")
    for metric, m in r["metrics"].items():
        verdict = {
            "vuot_noise_floor": "VUOT noise floor",
            "tut_ro_rang": "TUT RO RANG",
            "trong_noise_floor": "trong noise floor (that nhung khong dang ke)",
            "khong_ket_luan_duoc": "KHONG KET LUAN DUOC (KTC qua rong)",
        }[m["verdict"]]
        print(
            f"  {metric:8s} {m['baseline']:.4f} -> {m['candidate']:.4f}"
            f"   delta {m['delta']:+.4f}"
        )
        print(
            f"           tot hon {m['better']:3d} | kem hon {m['worse']:3d}"
            f" | n={m['n']}"
        )
        print(
            f"           KTC 95% [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
            f"  p={m['p_value']:.4f}  {verdict}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=0.008,
        help="Pre-registered e2e noise floor; a CI inside it is not adoptable.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = compare(
        load_per_case(args.baseline),
        load_per_case(args.candidate),
        metrics=tuple(args.metrics),
        resamples=args.resamples,
        seed=args.seed,
        noise_floor=args.noise_floor,
    )
    result["baseline_path"] = args.baseline.as_posix()
    result["candidate_path"] = args.candidate.as_posix()
    _render(args.candidate.name, result)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nda ghi {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
