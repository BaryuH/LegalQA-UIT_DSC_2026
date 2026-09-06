"""Rank failure groups by the score they actually cost, not by how many there are.

Twice in this project a fix was prioritised because its failure mode was
frequent, and twice the measurement said the mode cost almost nothing.
Wrong-document attribution covers 54% of cases and sits 0.022 below the
matched group; answers citing no source at all cover 13% and sit 0.099 below
it. Frequency ranked those the wrong way round.

The quantity that ranks them correctly is the deficit a group holds,
sum(1 - metric) over its cases, which is what could in principle be won back.
A small group of badly failing cases can hold more of it than a large group of
near-misses, and only the deficit shows that.

Groups come from a JSON mapping {group_name: [case ids]} - the output of
diagnose_citation_mismatch.py, or any hand-built partition. Without one the
report falls back to quartiles. Prints IDs and numbers only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean

_PAIRED = Path(__file__).with_name("compare_metrics_paired.py")
_spec = importlib.util.spec_from_file_location("_paired", _PAIRED)
assert _spec and _spec.loader
_paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paired)


def deficit_by_group(
    per_case: dict[str, dict],
    groups: dict[str, list[str]] | None,
    *,
    metric: str = "meteor",
) -> dict:
    scored = {
        cid: row[metric]
        for cid, row in per_case.items()
        if row.get(metric) is not None
    }
    total_deficit = sum(1.0 - v for v in scored.values())

    if groups:
        assigned = {cid for ids in groups.values() for cid in ids}
        buckets = {name: [i for i in ids if i in scored] for name, ids in groups.items()}
        rest = [cid for cid in scored if cid not in assigned]
        if rest:
            buckets["(con lai)"] = rest
    else:
        ordered = sorted(scored, key=lambda c: scored[c])
        n = len(ordered)
        buckets = {
            "tu phan vi 1 (te nhat)": ordered[: n // 4],
            "tu phan vi 2": ordered[n // 4 : n // 2],
            "tu phan vi 3": ordered[n // 2 : 3 * n // 4],
            "tu phan vi 4 (tot nhat)": ordered[3 * n // 4 :],
        }

    corpus_mean = mean(scored.values()) if scored else 0.0
    rows = []
    for name, ids in buckets.items():
        if not ids:
            continue
        deficit = sum(1.0 - scored[i] for i in ids)
        rows.append({
            "group": name,
            "n": len(ids),
            "share_of_cases": len(ids) / len(scored),
            "mean_metric": mean(scored[i] for i in ids),
            "deficit": deficit,
            "share_of_deficit": deficit / total_deficit if total_deficit else 0.0,
            # What lifting this group's MEAN to the corpus mean would be
            # worth globally. Summing each below-mean case instead rewards
            # group size: a group covering half the corpus scores highly on
            # that even when its own mean sits above the corpus mean, which
            # is not a gain anyone can collect.
            "ceiling_if_lifted_to_mean": (
                len(ids)
                * max(0.0, corpus_mean - mean(scored[i] for i in ids))
                / len(scored)
            ),
            # Kept for contrast: the sum of every below-mean shortfall inside
            # the group. Always at least as large, and misleading on its own
            # for a group that mixes strong and weak cases.
            "within_group_shortfall": (
                sum(max(0.0, corpus_mean - scored[i]) for i in ids) / len(scored)
            ),
            "worst_ids": sorted(ids, key=lambda c: scored[c])[:8],
        })
    # Sorted by the ceiling, not by raw deficit. A large group sitting at the
    # corpus mean holds a lot of deficit and offers nothing to win back;
    # ranking on deficit alone puts exactly that group first.
    rows.sort(key=lambda r: -r["ceiling_if_lifted_to_mean"])
    return {
        "metric": metric,
        "cases": len(scored),
        "mean_metric": mean(scored.values()) if scored else 0.0,
        "total_deficit": total_deficit,
        "groups": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--groups", type=Path, default=None,
                    help='JSON {"ten nhom": [case ids]}')
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Group by the error_type column of a classified report")
    ap.add_argument("--metric", default="meteor")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    per_case = _paired.load_per_case(args.metrics)
    groups: dict[str, list[str]] | None = None
    if args.groups:
        groups = json.loads(args.groups.read_text(encoding="utf-8"))
    elif args.from_csv:
        groups = {}
        for row in csv.DictReader(args.from_csv.open(encoding="utf-8")):
            groups.setdefault(row.get("error_type", "?"), []).append(row["id"])

    report = deficit_by_group(per_case, groups, metric=args.metric)
    print(f"{report['cases']} case | {report['metric']} trung binh "
          f"{report['mean_metric']:.4f} | tong tham hut {report['total_deficit']:.1f}\n")
    print(f"{'nhom':42s} {'n':>4s} {'%case':>7s} {'tb':>7s} "
          f"{'%tham hut':>10s} {'tran':>8s} {'trong nhom':>11s}")
    for r in report["groups"]:
        print(f"{r['group'][:42]:42s} {r['n']:4d} {r['share_of_cases']:7.1%} "
              f"{r['mean_metric']:7.4f} {r['share_of_deficit']:10.1%} "
              f"{r['ceiling_if_lifted_to_mean']:+8.4f} "
              f"{r['within_group_shortfall']:+11.4f}")
    print("\nID te nhat moi nhom:")
    for r in report["groups"]:
        print(f"  {r['group'][:36]:36s} {', '.join(r['worst_ids'])}")
    print("\n'tran'       = keo TRUNG BINH NHOM len trung binh corpus. Nhom da o"
          " tren trung binh thi bang 0."
          "\n'trong nhom' = cong thieu hut cua tung case duoi trung binh. Luon lon"
          " hon, va thuong lon chi vi nhom dong."
          "\nCa hai deu la tran tren va mang tinh tuong quan, khong phai loi hua.")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nda ghi {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
